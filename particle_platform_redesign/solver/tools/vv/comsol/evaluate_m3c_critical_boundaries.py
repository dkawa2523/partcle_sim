"""Evaluate the shared COMSOL/public-API critical-boundary microcase."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    GeometryData,
    RealizedTableSource,
    write,
)

TOOL_REVISION: Final = "m3c_critical_boundaries_evaluator_v1"
RECEIPT_PREFIX: Final = "M3CCB|configuration|"
ACTIVE_STATUS_CODE: Final = 1
STATE_COLUMNS: Final = (
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "current_status_code",
    "final_status_code",
)
GATE_COLUMNS: Final = (
    "producer",
    "step_label",
    "scenario_id",
    "gate",
    "status",
    "observed_value",
    "limit_value",
    "unit",
    "detail",
)


@dataclass(frozen=True, slots=True)
class Trajectory:
    particle_id: int
    time_s: np.ndarray[Any, np.dtype[np.float64]]
    position_m: np.ndarray[Any, np.dtype[np.float64]]
    velocity_m_per_s: np.ndarray[Any, np.dtype[np.float64]]
    current_status: np.ndarray[Any, np.dtype[np.int64]]
    final_status: np.ndarray[Any, np.dtype[np.int64]]


@dataclass(frozen=True, slots=True)
class StepComparison:
    raw_sha256: str
    comsol_metrics: dict[str, dict[str, Any]]
    candidate_metrics: dict[str, dict[str, Any]]
    candidate_summary: dict[str, Any]
    cross_solver_metrics: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(value: object, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must be a list of mappings")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    config = _mapping(json.loads(path.read_text(encoding="utf-8")), "configuration")
    required = {
        "schema_version",
        "evaluation_id",
        "evaluation_revision",
        "classification",
        "expected_comsol_version",
        "model",
        "case",
        "particles",
        "comsol_raw_export",
        "acceptance",
        "scope",
    }
    if set(config) != required:
        raise ValueError(f"configuration keys differ: {sorted(set(config) ^ required)}")
    if config["schema_version"] != 1 or config["evaluation_revision"] != 1:
        raise ValueError("unsupported critical-boundary configuration revision")
    case = _mapping(config["case"], "case")
    particles = _sequence(config["particles"], "particles")
    steps = _sequence(case.get("fixed_rk4_steps"), "case.fixed_rk4_steps")
    if (
        config["model"].get("source") != "from_scratch"
        or case.get("motion_mode") != "axisymmetric_rz_meridional"
        or case.get("output_frames") != 25
        or len(steps) != 3
        or [item.get("particle_id") for item in particles] != [1, 2, 3]
        or [item.get("path_kind") for item in particles]
        != ["free_flight", "outer_wall_specular", "axis_fold"]
    ):
        raise ValueError("critical-boundary case identity differs")
    raw = _mapping(config["comsol_raw_export"], "comsol_raw_export")
    if tuple(raw.get("columns", ())) != STATE_COLUMNS or raw.get("columns_per_frame") != 8:
        raise ValueError("COMSOL raw export contract differs")
    return config


def _steps(config: dict[str, Any]) -> list[dict[str, Any]]:
    return _sequence(config["case"]["fixed_rk4_steps"], "fixed_rk4_steps")


def _particles(config: dict[str, Any]) -> list[dict[str, Any]]:
    return _sequence(config["particles"], "particles")


def _parse_receipt(line: str) -> tuple[str, dict[str, str]] | None:
    marker = "M3CCB|"
    offset = line.find(marker)
    if offset < 0:
        return None
    fields = line[offset:].strip().split("|")
    values: dict[str, str] = {}
    for field in fields[2:]:
        if "=" not in field:
            raise ValueError(f"malformed COMSOL receipt field: {field}")
        key, value = field.split("=", 1)
        values[key] = value
    return fields[1], values


def _receipt_groups(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    log_path = root / "comsol_process.log"
    parsed = [
        receipt
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if (receipt := _parse_receipt(line)) is not None
    ]
    errors = [values for kind, values in parsed if kind == "run_error"]
    if errors:
        raise ValueError(f"COMSOL emitted run_error: {errors}")
    configurations = [values for kind, values in parsed if kind == "configuration"]
    solve_passes = [values for kind, values in parsed if kind == "solve_pass"]
    run_passes = [values for kind, values in parsed if kind == "run_pass"]
    if len(configurations) != 3 or len(solve_passes) != 3 or len(run_passes) != 1:
        raise ValueError("expected exactly three configuration/solve receipts and one run_pass")
    return configurations, solve_passes


def _required_configuration_receipt(config: dict[str, Any]) -> dict[str, str]:
    model = _mapping(config["model"], "model")
    return {
        "geometry": "axisymmetric_rectangle_10mm_by_20mm",
        "axis_boundary": f"[{model['axis_boundary_id']}]",
        "axis_feature_type": str(model["axis_feature_type"]),
        "axis_condition": str(model["axis_condition"]),
        "wall_boundaries": "[2, 3, 4]",
        "wall_condition": str(model["wall_condition"]),
        "release_groups": "surface,reflect,axis",
        "force_free": "true",
        "dynamic_charge_active": "false",
        "integrator": "classical_rk4",
        "model_source": "from_scratch",
        "model_saved": "false",
    }


def _validate_configuration_receipts(
    configurations: list[dict[str, str]], config: dict[str, Any]
) -> None:
    expected_steps = {float(item["seconds"]) for item in _steps(config)}
    observed_steps = {float(item["step_s"]) for item in configurations}
    if observed_steps != expected_steps:
        raise ValueError("COMSOL configuration step set differs")
    required = _required_configuration_receipt(config)
    for receipt in configurations:
        differences = {
            key: (receipt.get(key), value)
            for key, value in required.items()
            if receipt.get(key) != value
        }
        if differences:
            raise ValueError(f"COMSOL configuration receipt differs: {differences}")


def _validate_solve_receipts(solve_passes: list[dict[str, str]]) -> None:
    if any(
        int(item.get("output_times", "0")) != 25 or int(item.get("particle_rows", "0")) != 3
        for item in solve_passes
    ):
        raise ValueError("COMSOL solve receipt has unexpected frame or particle count")


def _validate_receipts(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    configurations, solve_passes = _receipt_groups(root)
    _validate_configuration_receipts(configurations, config)
    _validate_solve_receipts(solve_passes)
    return {
        "configuration_count": len(configurations),
        "solve_pass_count": len(solve_passes),
        "run_pass_count": 1,
        "validated": True,
    }


def _read_wide(path: Path, config: dict[str, Any]) -> dict[int, Trajectory]:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for raw in csv.reader(stream):
            if not raw or raw[0].lstrip().startswith("%"):
                continue
            rows.append([float(value) for value in raw])
    frames = int(config["case"]["output_frames"])
    width = len(STATE_COLUMNS)
    if len(rows) != 3 or any(len(row) != frames * width for row in rows):
        raise ValueError(f"{path}: expected three {frames * width}-value particle rows")
    expected_times = np.arange(frames, dtype=np.float64) * float(
        config["case"]["output_interval_s"]
    )
    trajectories: dict[int, Trajectory] = {}
    for row in rows:
        values = np.asarray(row, dtype=np.float64).reshape(frames, width)
        ids = values[:, 0].astype(np.int64)
        particle_id = int(ids[0])
        if not np.all(ids == particle_id) or particle_id in trajectories:
            raise ValueError(f"{path}: invalid or duplicate particle id")
        if not np.allclose(values[:, 1], expected_times, rtol=0.0, atol=2e-15):
            raise ValueError(f"{path}: output time grid differs")
        trajectories[particle_id] = Trajectory(
            particle_id,
            values[:, 1].copy(),
            values[:, 2:4].copy(),
            values[:, 4:6].copy(),
            values[:, 6].astype(np.int64),
            values[:, 7].astype(np.int64),
        )
    if set(trajectories) != {1, 2, 3}:
        raise ValueError(f"{path}: particle ids differ")
    return trajectories


def _analytic(
    particle: dict[str, Any], times: np.ndarray[Any, np.dtype[np.float64]], radius_m: float
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    position0 = np.asarray(particle["source_position_m"], dtype=np.float64)
    velocity0 = np.asarray(particle["source_velocity_m_per_s"], dtype=np.float64)
    position = position0[None, :] + times[:, None] * velocity0[None, :]
    velocity = np.repeat(velocity0[None, :], times.size, axis=0)
    event_time = particle.get("analytic_event_time_s")
    event_time_s = float(event_time) if event_time is not None else 0.0
    after = times > event_time_s if event_time is not None else np.zeros(times.size, dtype=bool)
    if particle["path_kind"] == "outer_wall_specular":
        residual = times[after] - event_time_s
        position[after, 0] = radius_m - abs(float(velocity0[0])) * residual
        velocity[after, 0] = -velocity0[0]
    elif particle["path_kind"] == "axis_fold":
        position[:, 0] = np.abs(position[:, 0])
        velocity[after, 0] = -velocity0[0]
    return position, velocity


def _event_time_from_first_post_state(
    trajectory: Trajectory, particle: dict[str, Any], radius_m: float
) -> tuple[float, float, float]:
    event_time = float(particle["analytic_event_time_s"])
    index = int(np.flatnonzero(trajectory.time_s > event_time)[0])
    time_s = float(trajectory.time_s[index])
    radial_speed = abs(float(trajectory.velocity_m_per_s[index, 0]))
    if particle["path_kind"] == "outer_wall_specular":
        travelled = radius_m - float(trajectory.position_m[index, 0])
    else:
        travelled = float(trajectory.position_m[index, 0])
    reconstructed = time_s - travelled / radial_speed
    return reconstructed, time_s, travelled


def _max_vector_error(actual: np.ndarray[Any, Any], expected: np.ndarray[Any, Any]) -> float:
    return float(np.max(np.linalg.norm(actual - expected, axis=1)))


def _trajectory_metrics(
    trajectory: Trajectory, particle: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    radius_m = float(config["model"]["radius_m"])
    expected_position, expected_velocity = _analytic(particle, trajectory.time_s, radius_m)
    metrics: dict[str, Any] = {
        "maximum_position_absolute_error_m": _max_vector_error(
            trajectory.position_m, expected_position
        ),
        "maximum_velocity_absolute_error_m_per_s": _max_vector_error(
            trajectory.velocity_m_per_s, expected_velocity
        ),
        "initial_position_absolute_error_m": float(
            np.linalg.norm(
                trajectory.position_m[0]
                - np.asarray(particle["source_position_m"], dtype=np.float64)
            )
        ),
        "all_frames_active": bool(np.all(trajectory.current_status == ACTIVE_STATUS_CODE)),
        "final_status_active": bool(np.all(trajectory.final_status == ACTIVE_STATUS_CODE)),
        "frames": int(trajectory.time_s.size),
    }
    if particle["path_kind"] == "free_flight":
        metrics["first_positive_time_position_m"] = trajectory.position_m[1].tolist()
        metrics["departed_into_domain"] = bool(trajectory.position_m[1, 1] > 0.0)
    else:
        observed, first_post_time, residual_distance = _event_time_from_first_post_state(
            trajectory, particle, radius_m
        )
        metrics.update(
            {
                "reconstructed_event_time_s": observed,
                "event_time_absolute_error_s": abs(
                    observed - float(particle["analytic_event_time_s"])
                ),
                "first_post_event_output_time_s": first_post_time,
                "observed_residual_distance_m": residual_distance,
                "expected_residual_distance_m": abs(
                    float(particle["analytic_post_velocity_m_per_s"][0])
                )
                * (first_post_time - float(particle["analytic_event_time_s"])),
            }
        )
    return metrics


def _gate(
    rows: list[dict[str, Any]],
    producer: str,
    step: str,
    scenario: str,
    name: str,
    passed: bool,
    observed: object,
    limit: object,
    unit: str,
    detail: str,
) -> None:
    rows.append(
        {
            "producer": producer,
            "step_label": step,
            "scenario_id": scenario,
            "gate": name,
            "status": "PASS" if passed else "FAIL",
            "observed_value": observed,
            "limit_value": limit,
            "unit": unit,
            "detail": detail,
        }
    )


def _add_path_gates(
    rows: list[dict[str, Any]],
    producer: str,
    step: str,
    particle: dict[str, Any],
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    acceptance = config["acceptance"]
    scenario = str(particle["id"])
    for name, key, limit_key, unit in (
        (
            "initial_state",
            "initial_position_absolute_error_m",
            "maximum_initial_absolute_error_m",
            "m",
        ),
        (
            "analytic_position_path",
            "maximum_position_absolute_error_m",
            "maximum_position_absolute_error_m",
            "m",
        ),
        (
            "analytic_velocity_path",
            "maximum_velocity_absolute_error_m_per_s",
            "maximum_velocity_absolute_error_m_per_s",
            "m/s",
        ),
    ):
        observed = float(metrics[key])
        limit = float(acceptance[limit_key])
        _gate(rows, producer, step, scenario, name, observed <= limit, observed, limit, unit, key)
    active = bool(metrics["all_frames_active"] and metrics["final_status_active"])
    _gate(
        rows,
        producer,
        step,
        scenario,
        "nonterminal_lifecycle",
        active,
        active,
        True,
        "boolean",
        "all saved current and final statuses remain active",
    )
    if particle["path_kind"] == "free_flight":
        _gate(
            rows,
            producer,
            step,
            scenario,
            "surface_departure",
            bool(metrics["departed_into_domain"]),
            metrics["first_positive_time_position_m"],
            "z > 0",
            "m",
            "first post-release frame is inside without a zero-time terminal state",
        )
    else:
        observed = float(metrics["event_time_absolute_error_s"])
        limit = float(acceptance["maximum_event_time_absolute_error_s"])
        _gate(
            rows,
            producer,
            step,
            scenario,
            "reconstructed_event_time",
            observed <= limit,
            observed,
            limit,
            "s absolute error",
            "event time reconstructed from the first post-event state and residual distance",
        )
        residual_error = abs(
            float(metrics["observed_residual_distance_m"])
            - float(metrics["expected_residual_distance_m"])
        )
        path_limit = float(acceptance["maximum_position_absolute_error_m"])
        _gate(
            rows,
            producer,
            step,
            scenario,
            "post_event_residual_flight",
            residual_error <= path_limit,
            residual_error,
            path_limit,
            "m",
            "first post-event frame includes the analytic residual flight",
        )


def _candidate_bundle(config: dict[str, Any], config_hash: str) -> DataBundle:
    radius = float(config["model"]["radius_m"])
    height = float(config["model"]["height_m"])
    nodes = np.asarray([[0.0, 0.0], [radius, 0.0], [radius, height], [0.0, height]])
    boundary = BoundaryData(
        line2=np.asarray([[0, 1], [1, 2], [2, 3]], dtype="<i8"),
        boundary_id=np.asarray([10, 20, 30], dtype="<i4"),
        group_id=np.asarray([0, 1, 2], dtype="<i4"),
        material_id=np.zeros(3, dtype="<i4"),
        owner_cell_type=np.full(3, 2, dtype="<u1"),
        owner_cell_local_index=np.zeros(3, dtype="<i8"),
        orientation=np.ones(3, dtype="<i1"),
        external_id=np.asarray([2, 4, 3], dtype="<i8"),
    )
    geometry = GeometryData(
        nodes_m=nodes.astype("<f8"),
        boundary=boundary,
        group_names=("lower_surface", "outer_wall", "top_cap"),
        quad4=np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        quad4_domain_id=np.asarray([0], dtype="<i4"),
    )
    table_particles = [
        particle for particle in _particles(config) if particle["source_kind"] == "table"
    ]
    diameter = float(config["case"]["particle_diameter_m"])
    source = RealizedTableSource(
        name="table_particles",
        particle_id=np.asarray([item["particle_id"] for item in table_particles], dtype="<i8"),
        release_time_s=np.zeros(len(table_particles), dtype="<f8"),
        position_m=np.asarray([item["source_position_m"] for item in table_particles], dtype="<f8"),
        velocity_m_s=np.asarray(
            [item["source_velocity_m_per_s"] for item in table_particles], dtype="<f8"
        ),
        charge_number=np.full(
            len(table_particles), float(config["case"]["source_charge_number_e"]), dtype="<f8"
        ),
        mass_kg=np.full(
            len(table_particles), float(config["case"]["particle_mass_kg"]), dtype="<f8"
        ),
        drag_diameter_m=np.full(len(table_particles), diameter, dtype="<f8"),
        electrostatic_radius_m=np.full(len(table_particles), 0.5 * diameter, dtype="<f8"),
        displaced_volume_m3=np.full(len(table_particles), math.pi * diameter**3 / 6.0, dtype="<f8"),
        model_weight=np.ones(len(table_particles), dtype="<f8"),
        material_id=np.zeros(len(table_particles), dtype="<i4"),
    )
    provenance = json.dumps(
        {
            "producer": "m3c-critical-boundaries-vv",
            "producer_version": TOOL_REVISION,
            "source_sha256": f"sha256:{config_hash}",
            "field_semantics_revision": "force_free_v1",
            "producer_metadata": {"evaluation_id": config["evaluation_id"]},
        }
    )
    return DataBundle("axisymmetric_rz", provenance, geometry, sources=(source,))


def _particle_properties(config: dict[str, Any]) -> dict[str, Any]:
    diameter = float(config["case"]["particle_diameter_m"])
    return {
        "charge_number": float(config["case"]["source_charge_number_e"]),
        "mass_kg": float(config["case"]["particle_mass_kg"]),
        "drag_diameter_m": diameter,
        "electrostatic_radius_m": 0.5 * diameter,
        "displaced_volume_m3": math.pi * diameter**3 / 6.0,
        "model_weight": 1.0,
        "material_id": 0,
    }


def _candidate_document(
    config: dict[str, Any], step: dict[str, Any], content_hash: str
) -> dict[str, Any]:
    surface = _particles(config)[0]
    frames = int(config["case"]["output_frames"])
    times = (
        np.arange(frames, dtype=np.float64) * float(config["case"]["output_interval_s"])
    ).tolist()
    return {
        "format_version": 2,
        "case": {
            "name": f"m3c_critical_boundaries_{step['label']}",
            "data_path": "candidate_input.h5",
            "expected_content_hash": content_hash,
        },
        "motion": {"mode": config["case"]["motion_mode"]},
        "time": {
            "start_s": 0.0,
            "end_s": float(config["case"]["time_end_s"]),
            "dt_s": float(step["seconds"]),
        },
        "solver": {
            "integrator": "rk4_fixed",
            "backend": "cpu",
            "seed": 0,
            "event": {
                "geometry_rtol": 1.0e-12,
                "roundoff_ulps": 64,
                "max_refinements": 48,
                "max_interactions_per_step": 8,
                "corner_policy": "priority_then_combined_normal_v1",
            },
        },
        "resources": {"memory_limit_mb": 128},
        "physics": {"charge": {"model": "fixed"}},
        "sources": [
            {
                "name": "surface_particle",
                "type": "surface",
                "boundary_group": "lower_surface",
                "count": 1,
                "particle_id_start": int(surface["particle_id"]),
                "particle": _particle_properties(config),
                "position": {
                    "model": "edge_fraction",
                    "fraction": float(surface["edge_fraction"]),
                },
                "velocity": {
                    "model": "fixed",
                    "value_m_s": surface["source_velocity_m_per_s"],
                },
                "release": {"model": "fixed", "time_s": 0.0},
            },
            {"name": "table_particles", "type": "table", "table": "table_particles"},
        ],
        "boundaries": [
            {"boundary_group": "lower_surface", "priority": 10, "law": "specular"},
            {"boundary_group": "outer_wall", "priority": 20, "law": "specular"},
            {"boundary_group": "top_cap", "priority": 30, "law": "escape"},
        ],
        "output": {
            "trajectories": {
                "selection": "all",
                "schedule": {"explicit_times_s": times},
            }
        },
    }


def _result_trajectories(result: Any, config: dict[str, Any]) -> dict[int, Trajectory]:
    buckets: dict[int, list[tuple[float, np.ndarray[Any, Any], np.ndarray[Any, Any], int]]] = {
        1: [],
        2: [],
        3: [],
    }
    for frame in result.iter_frames():
        for index, particle_id_value in enumerate(frame.particle_id):
            particle_id = int(particle_id_value)
            buckets[particle_id].append(
                (
                    float(frame.time_s),
                    frame.position_m[index].copy(),
                    frame.velocity_m_s[index].copy(),
                    int(frame.lifecycle[index]),
                )
            )
    trajectories: dict[int, Trajectory] = {}
    for particle_id, rows in buckets.items():
        if len(rows) != int(config["case"]["output_frames"]):
            raise ValueError(f"candidate particle {particle_id} has incomplete frame history")
        trajectories[particle_id] = Trajectory(
            particle_id,
            np.asarray([row[0] for row in rows], dtype=np.float64),
            np.asarray([row[1] for row in rows], dtype=np.float64),
            np.asarray([row[2] for row in rows], dtype=np.float64),
            np.asarray([ACTIVE_STATUS_CODE if row[3] == 1 else row[3] for row in rows]),
            np.full(len(rows), ACTIVE_STATUS_CODE, dtype=np.int64),
        )
    return trajectories


def _run_candidate(
    root: Path, config: dict[str, Any], step: dict[str, Any], config_hash: str
) -> tuple[dict[int, Trajectory], dict[str, Any]]:
    step_root = root / str(step["label"])
    step_root.mkdir(parents=True)
    data_info = write(step_root / "candidate_input.h5", _candidate_bundle(config, config_hash))
    case_path = step_root / "candidate_case.yaml"
    case_path.write_text(
        yaml.safe_dump(_candidate_document(config, step, data_info.content_hash), sort_keys=False),
        encoding="utf-8",
    )
    result_path = step_root / "result"
    simulate(load_case(case_path), result_path)
    result = open_result(result_path)
    release = result.read_release_events()
    boundary = result.read_boundary_events()
    failures = result.read_failure_events()
    manifest = dict(result.manifest)
    interactions = _mapping(manifest["boundary_interactions"], "boundary_interactions")
    summary = {
        "release_particle_ids": release.particle_id.astype(int).tolist(),
        "boundary_event_particle_ids": boundary.particle_id.astype(int).tolist(),
        "boundary_event_times_s": boundary.time_s.tolist(),
        "boundary_event_positions_m": boundary.position_m.tolist(),
        "boundary_event_pre_velocity_m_per_s": boundary.velocity_pre_m_s.tolist(),
        "boundary_event_post_velocity_m_per_s": boundary.velocity_post_m_s.tolist(),
        "boundary_event_laws": boundary.law_id.tolist(),
        "boundary_event_outcomes": boundary.outcome.tolist(),
        "failure_count": int(failures.particle_id.size),
        "axis_crossings": int(interactions["axis_crossings"]),
        "wall_events": int(interactions["wall_events"]),
        "case_sha256": _sha256(case_path),
        "canonical_input_sha256": _sha256(step_root / "candidate_input.h5"),
        "result_manifest_sha256": _sha256(result_path / "run.json"),
        "revisions": {
            key: manifest.get(key)
            for key in (
                "case_schema_version",
                "result_schema_version",
                "engine_algorithm_revision",
                "source_algorithm_revision",
                "event_algorithm_revision",
                "boundary_algorithm_revision",
                "result_algorithm_revision",
            )
        },
    }
    return _result_trajectories(result, config), summary


def _add_candidate_semantic_gates(
    rows: list[dict[str, Any]], step: str, summary: dict[str, Any], config: dict[str, Any]
) -> None:
    acceptance = config["acceptance"]
    reflection = _particles(config)[1]
    expected_event_time = float(reflection["analytic_event_time_s"])
    ids = summary["boundary_event_particle_ids"]
    event_ok = (
        ids == [2]
        and summary["boundary_event_laws"] == ["specular"]
        and summary["boundary_event_outcomes"] == ["reflected"]
    )
    _gate(
        rows,
        "candidate",
        step,
        "specular_reflection",
        "single_specular_event",
        event_ok,
        {"ids": ids, "laws": summary["boundary_event_laws"]},
        {"ids": [2], "laws": ["specular"]},
        "event identity",
        "surface departure and axis passage emit no material-wall event",
    )
    time_error = (
        abs(float(summary["boundary_event_times_s"][0]) - expected_event_time)
        if event_ok
        else math.inf
    )
    _gate(
        rows,
        "candidate",
        step,
        "specular_reflection",
        "direct_event_time",
        time_error <= float(acceptance["maximum_event_time_absolute_error_s"]),
        time_error,
        acceptance["maximum_event_time_absolute_error_s"],
        "s absolute error",
        "candidate public event row versus analytic impact time",
    )
    axis_ok = summary["axis_crossings"] == 1 and 3 not in ids
    _gate(
        rows,
        "candidate",
        step,
        "axis_crossing",
        "axis_is_not_material_wall",
        axis_ok,
        {"axis_crossings": summary["axis_crossings"], "wall_event_ids": ids},
        {"axis_crossings": 1, "wall_event_ids_excludes": 3},
        "counts",
        "axis passage is counted separately and has no wall event row",
    )
    failures_ok = summary["failure_count"] == 0
    _gate(
        rows,
        "candidate",
        step,
        "all",
        "zero_failures",
        failures_ok,
        summary["failure_count"],
        0,
        "events",
        "public result has no failure event",
    )


def _add_cross_solver_gates(
    rows: list[dict[str, Any]],
    step: str,
    comsol: dict[int, Trajectory],
    candidate: dict[int, Trajectory],
    config: dict[str, Any],
) -> dict[str, Any]:
    position_limit = float(config["acceptance"]["maximum_cross_solver_position_difference_m"])
    velocity_limit = float(config["acceptance"]["maximum_cross_solver_velocity_difference_m_per_s"])
    summary: dict[str, Any] = {}
    for particle in _particles(config):
        particle_id = int(particle["particle_id"])
        scenario = str(particle["id"])
        left = comsol[particle_id]
        right = candidate[particle_id]
        if not np.array_equal(left.time_s, right.time_s):
            raise ValueError(f"{step}/{scenario}: producer time grids differ")
        position_error = _max_vector_error(left.position_m, right.position_m)
        velocity_error = _max_vector_error(left.velocity_m_per_s, right.velocity_m_per_s)
        _gate(
            rows,
            "cross_solver",
            step,
            scenario,
            "position_path",
            position_error <= position_limit,
            position_error,
            position_limit,
            "m",
            "COMSOL and public-API trajectories on the common saved grid",
        )
        _gate(
            rows,
            "cross_solver",
            step,
            scenario,
            "velocity_path",
            velocity_error <= velocity_limit,
            velocity_error,
            velocity_limit,
            "m/s",
            "COMSOL and public-API velocities on the common saved grid",
        )
        summary[scenario] = {
            "maximum_position_difference_m": position_error,
            "maximum_velocity_difference_m_per_s": velocity_error,
        }
    return summary


def _measure_producer(
    rows: list[dict[str, Any]],
    producer: str,
    step_label: str,
    trajectories: dict[int, Trajectory],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    metrics_by_scenario: dict[str, dict[str, Any]] = {}
    for particle in _particles(config):
        particle_id = int(particle["particle_id"])
        scenario = str(particle["id"])
        metrics = _trajectory_metrics(trajectories[particle_id], particle, config)
        metrics_by_scenario[scenario] = metrics
        _add_path_gates(rows, producer, step_label, particle, metrics, config)
    return metrics_by_scenario


def _evaluate_step(
    root: Path,
    candidate_root: Path,
    config: dict[str, Any],
    step: dict[str, Any],
    config_hash: str,
    gates: list[dict[str, Any]],
) -> StepComparison:
    label = str(step["label"])
    raw_path = root / label / str(config["comsol_raw_export"]["table"])
    comsol = _read_wide(raw_path, config)
    candidate, candidate_summary = _run_candidate(candidate_root, config, step, config_hash)
    comsol_metrics = _measure_producer(gates, "comsol", label, comsol, config)
    candidate_metrics = _measure_producer(gates, "candidate", label, candidate, config)
    _add_candidate_semantic_gates(gates, label, candidate_summary, config)
    cross_solver_metrics = _add_cross_solver_gates(gates, label, comsol, candidate, config)
    return StepComparison(
        raw_sha256=_sha256(raw_path),
        comsol_metrics=comsol_metrics,
        candidate_metrics=candidate_metrics,
        candidate_summary=candidate_summary,
        cross_solver_metrics=cross_solver_metrics,
    )


def _maximum_metric(comparisons: dict[str, StepComparison], attribute: str, metric: str) -> float:
    values = (
        float(item[metric])
        for comparison in comparisons.values()
        for item in getattr(comparison, attribute).values()
    )
    return max(values)


def _comparison_maxima(comparisons: dict[str, StepComparison]) -> dict[str, float]:
    return {
        "comsol_position_error_m": _maximum_metric(
            comparisons, "comsol_metrics", "maximum_position_absolute_error_m"
        ),
        "comsol_velocity_error_m_per_s": _maximum_metric(
            comparisons, "comsol_metrics", "maximum_velocity_absolute_error_m_per_s"
        ),
        "candidate_position_error_m": _maximum_metric(
            comparisons, "candidate_metrics", "maximum_position_absolute_error_m"
        ),
        "candidate_velocity_error_m_per_s": _maximum_metric(
            comparisons, "candidate_metrics", "maximum_velocity_absolute_error_m_per_s"
        ),
        "cross_solver_position_difference_m": _maximum_metric(
            comparisons, "cross_solver_metrics", "maximum_position_difference_m"
        ),
        "cross_solver_velocity_difference_m_per_s": _maximum_metric(
            comparisons, "cross_solver_metrics", "maximum_velocity_difference_m_per_s"
        ),
    }


def _optional_provenance(root: Path) -> dict[str, Any] | None:
    path = root / "provenance.json"
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else None


def _build_report(
    root: Path,
    config: dict[str, Any],
    config_hash: str,
    receipts: dict[str, Any],
    comparisons: dict[str, StepComparison],
    gates: list[dict[str, Any]],
) -> dict[str, Any]:
    failure_count = sum(row["status"] == "FAIL" for row in gates)
    pass_count = sum(row["status"] == "PASS" for row in gates)
    return {
        "schema_version": 1,
        "evaluation_id": config["evaluation_id"],
        "evaluation_revision": config["evaluation_revision"],
        "evaluation_status": "COMPLETE",
        "scientific_status": "PASS" if failure_count == 0 else "FAIL",
        "tool_revision": TOOL_REVISION,
        "gate_counts": {"pass": pass_count, "fail": failure_count},
        "configuration_sha256": config_hash,
        "configuration_receipts": receipts,
        "comsol_provenance": _optional_provenance(root),
        "comsol_raw_sha256": {
            label: comparison.raw_sha256 for label, comparison in comparisons.items()
        },
        "comsol_metrics": {
            label: comparison.comsol_metrics for label, comparison in comparisons.items()
        },
        "candidate_metrics": {
            label: comparison.candidate_metrics for label, comparison in comparisons.items()
        },
        "candidate_runs": {
            label: comparison.candidate_summary for label, comparison in comparisons.items()
        },
        "cross_solver_metrics": {
            label: comparison.cross_solver_metrics for label, comparison in comparisons.items()
        },
        "maxima": _comparison_maxima(comparisons),
        "scope": config["scope"],
        "claim_separation": {
            "surface_departure": "TESTED",
            "specular_reflection_and_residual_flight": "TESTED",
            "rz_axis_non_wall_continuation": "TESTED",
            "terminal_boundary_semantics": "REUSED_FROM_M3C0_BOUNDARY_SEMANTICS_V2",
            "grazing_corner_multiple_hit_probabilistic": "NOT_TESTED",
            "force_field_native_representation": "NOT_APPLICABLE_TO_FORCE_FREE_MICROCASE",
            "comsol_is_golden_truth": False,
        },
    }


def _publish_evidence(evidence: Path, report: dict[str, Any], gates: list[dict[str, Any]]) -> None:
    evidence.mkdir(parents=True)
    with (evidence / "comparison_manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    _write_gates(evidence / "gates.csv", gates)
    _write_readme(evidence / "README.md", report)


def _write_gates(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=GATE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "observed_value": json.dumps(row["observed_value"], sort_keys=True),
                    "limit_value": json.dumps(row["limit_value"], sort_keys=True),
                }
            )


def _write_readme(path: Path, report: dict[str, Any]) -> None:
    maxima = report["maxima"]
    text = f"""# M3-C0 critical 2-D boundaries v1

Evaluation status: **{report["evaluation_status"]}**. Scientific decision:
**{report["scientific_status"]}**. All {report["gate_counts"]["pass"]} registered
gates pass with {report["gate_counts"]["fail"]} failures.

One from-scratch 10 mm by 20 mm axisymmetric rectangle carries three force-free
particles at the same initial states in COMSOL 6.4 and the production public
API. The three fixed RK4 steps are 1, 0.5, and 0.25 ms; 25 common frames cover
0--6 ms.

| scenario | accepted behavior |
|---|---|
| surface departure | starts at `(r,z)=(6 mm,0)` and remains active while moving into the domain |
| specular reflection | hits `r=10 mm` at 1.95 ms, reverses only radial velocity, and advances the 0.05 ms residual to the first post-hit frame |
| R-Z axis passage | reaches `r=0` at 1.95 ms, stays active, reverses chart radial velocity, and emits no material-wall event |

Maximum COMSOL analytic position/velocity errors are
`{maxima["comsol_position_error_m"]:.6g} m` and
`{maxima["comsol_velocity_error_m_per_s"]:.6g} m/s`. Maximum production-API
analytic errors are `{maxima["candidate_position_error_m"]:.6g} m` and
`{maxima["candidate_velocity_error_m_per_s"]:.6g} m/s`. The maximum direct
COMSOL/API differences are `{maxima["cross_solver_position_difference_m"]:.6g} m`
and `{maxima["cross_solver_velocity_difference_m_per_s"]:.6g} m/s`.

COMSOL boundary 1 is owned by `AxialSymmetry` and is absent from the ordinary
wall selection `[2,3,4]`. Its nonterminal `Bounce` setting is the axisymmetric
chart continuation, not a material-wall event. The production result likewise
reports one axis crossing and no boundary event for particle 3.

This evidence does not certify grazing/corners, multiple material hits,
probabilistic laws, finite-radius contact, forces, fields, or native-field
equivalence. Exact gates and provenance are in `gates.csv` and
`comparison_manifest.json`.
"""
    path.write_text(text, encoding="utf-8")


def evaluate(root: Path, config_path: Path, evidence: Path) -> dict[str, Any]:
    """Run the public candidate, evaluate both producers, and publish compact evidence."""

    root = root.resolve()
    config_path = config_path.resolve()
    evidence = evidence.resolve()
    if evidence.exists():
        raise FileExistsError(f"evidence already exists: {evidence}")
    candidate_root = root / "candidate"
    if candidate_root.exists():
        raise FileExistsError(f"candidate output already exists: {candidate_root}")
    config = _load_config(config_path)
    config_hash = _sha256(config_path)
    receipts = _validate_receipts(root, config)
    gates: list[dict[str, Any]] = []
    comparisons: dict[str, StepComparison] = {}
    candidate_root.mkdir()

    for step in _steps(config):
        label = str(step["label"])
        comparisons[label] = _evaluate_step(root, candidate_root, config, step, config_hash, gates)

    report = _build_report(root, config, config_hash, receipts, comparisons, gates)
    _publish_evidence(evidence, report, gates)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    arguments = parser.parse_args()
    report = evaluate(arguments.output_root, arguments.config, arguments.evidence)
    print(json.dumps(report["gate_counts"], sort_keys=True))
    return 0 if report["scientific_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
