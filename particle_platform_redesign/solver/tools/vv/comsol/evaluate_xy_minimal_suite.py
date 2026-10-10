"""Evaluate fresh COMSOL Cartesian XY microcases through the public solver API.

The COMSOL runner creates deliberately small, unsaved models.  This evaluator
normalizes their time histories, builds an independently specified candidate
case, and compares both solvers with closed-form references.  It is external
V&V tooling: no COMSOL-specific branch is imported by the solver core.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal

import numpy as np
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    FieldData,
    GeometryData,
    RealizedSurfaceSource,
    RealizedTableSource,
    RegularLayout,
    write,
)

TOOL_REVISION: Final = "comsol_xy_minimal_suite_v1"
SCENARIOS: Final = (
    "ballistic",
    "electric",
    "linear_drag",
    "surface_departure",
    "specular",
    "stick",
    "hold",
)
STEPS_S: Final = (0.02, 0.01, 0.005)
OUTPUT_TIMES_S: Final = np.linspace(0.0, 0.5, 11, dtype=np.float64)
EVENT_TIME_S: Final = 0.25
ELEMENTARY_CHARGE_C: Final = 1.602176634e-19
ELECTRIC_FIELD_X_V_M: Final = 2.4966036297843054e6
DRAG_RATE_S_INV: Final = 2.0
DRAG_GAS_VELOCITY_M_S: Final = np.asarray([0.1, 0.05], dtype=np.float64)
DRAG_DIAMETER_M: Final = 2.0e-6
DRAG_MASS_KG: Final = 4.0e-15
DRAG_MEAN_FREE_PATH_M: Final = 1.0e-6


@dataclass(frozen=True, slots=True)
class Trajectory:
    time_s: np.ndarray
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    status: np.ndarray


@dataclass(frozen=True, slots=True)
class ScenarioMetrics:
    scenario: str
    step_s: float
    maximum_cross_position_error_m: float
    maximum_cross_velocity_error_m_s: float
    maximum_comsol_analytic_position_error_m: float
    maximum_comsol_analytic_velocity_error_m_s: float
    maximum_candidate_analytic_position_error_m: float
    maximum_candidate_analytic_velocity_error_m_s: float
    compared_frame_count: int
    candidate_boundary_event_count: int
    semantic_gate: bool
    numeric_gate: bool


@dataclass(frozen=True, slots=True)
class Lineage:
    receipt_sha256: str
    comsol_version: str
    compiled_class_sha256: str
    source_hashes: dict[str, str]
    configuration_record_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _step_directory(step_s: float) -> str:
    return f"dt_{step_s:.3e}".replace("+", "_")


def _configuration_expectation(scenario: str) -> dict[str, str]:
    return {
        "coordinate_system": "cartesian_xy",
        "formulation": "NewtonianFirstOrder",
        "force": (
            "constant_qE"
            if scenario == "electric"
            else "linear_relaxation"
            if scenario == "linear_drag"
            else "none"
        ),
        "wall": {
            "stick": "Stick",
            "hold": "Freeze",
            "specular": "Bounce",
        }.get(scenario, "Bounce"),
        "release": "exact_surface_origin" if scenario == "surface_departure" else "interior",
        "model_saved": "false",
    }


def _parse_xymin_line(line: str, path: Path) -> dict[str, str] | None:
    fields = line.strip().split("|")
    if len(fields) < 2 or fields[0] != "XYMIN":
        return None
    record: dict[str, str] = {"kind": fields[1]}
    for value in fields[2:]:
        key, separator, item = value.partition("=")
        if not separator or not key:
            raise ValueError(f"invalid configuration receipt field in {path}: {value!r}")
        record[key] = item
    return record


def _validate_configuration_records(
    records: list[dict[str, str]],
    *,
    run_pass: bool,
) -> None:
    if not run_pass:
        raise ValueError("COMSOL configuration log has no successful no-save completion record")
    expected_pairs = {(scenario, step_s) for scenario in SCENARIOS for step_s in STEPS_S}
    actual_pairs: set[tuple[str, float]] = set()
    for record in records:
        scenario = record.get("scenario", "")
        step_s = float(record.get("step_s", "nan"))
        if scenario not in SCENARIOS:
            raise ValueError(f"unexpected COMSOL scenario in receipt: {scenario}")
        for key, expected in _configuration_expectation(scenario).items():
            if record.get(key) != expected:
                raise ValueError(
                    f"COMSOL configuration differs for {scenario}: {key}={record.get(key)!r}"
                )
        actual_pairs.add((scenario, step_s))
    if actual_pairs != expected_pairs or len(records) != len(expected_pairs):
        raise ValueError(
            "COMSOL configuration receipt does not cover the locked scenario/step matrix"
        )


def _parse_configuration_log(path: Path) -> list[dict[str, str]]:
    parsed = [
        record
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if (record := _parse_xymin_line(line, path)) is not None
    ]
    records = [record for record in parsed if record["kind"] == "configuration"]
    run_pass = any(
        record["kind"] == "run_pass"
        and record.get("scenario_count") == "7"
        and record.get("model_saved") == "false"
        for record in parsed
    )
    _validate_configuration_records(records, run_pass=run_pass)
    return records


def _validate_registered_sources(
    receipt_root: Path,
    source_records: object,
) -> dict[str, str]:
    current_sources = {
        "java": Path(__file__).resolve().parent / "comsol" / "RunXYMinimalSuite.java",
        "evaluator": Path(__file__).resolve(),
        "runner": Path(__file__).resolve().parent / "run_xy_minimal_suite.ps1",
    }
    if not isinstance(source_records, dict) or set(source_records) != set(current_sources):
        raise ValueError("run receipt has an incomplete registered-source set")
    source_hashes: dict[str, str] = {}
    for name, current_path in current_sources.items():
        record = source_records[name]
        if not isinstance(record, dict):
            raise ValueError(f"invalid registered-source record: {name}")
        digest = str(record.get("sha256", "")).lower()
        staged_relative = record.get("path")
        if not isinstance(staged_relative, str) or not digest:
            raise ValueError(f"invalid registered-source fields: {name}")
        staged_path = (receipt_root / staged_relative).resolve()
        if _sha256(current_path) != digest or _sha256(staged_path) != digest:
            raise ValueError(f"registered source lineage differs for {name}")
        source_hashes[name] = digest
    return source_hashes


def _validate_raw_artifacts(raw_root: Path, raw_records: object) -> None:
    if not isinstance(raw_records, list):
        raise ValueError("run receipt has no raw-artifact inventory")
    expected_raw = {
        f"{scenario}/{_step_directory(step_s)}/state_raw_wide.csv"
        for scenario in SCENARIOS
        for step_s in STEPS_S
    }
    recorded_raw: set[str] = set()
    for record in raw_records:
        if not isinstance(record, dict):
            raise ValueError("invalid raw-artifact record")
        relative = str(record.get("path", "")).replace("\\", "/")
        digest = str(record.get("sha256", "")).lower()
        raw_path = raw_root / relative
        if relative in recorded_raw or _sha256(raw_path) != digest:
            raise ValueError(f"raw COMSOL artifact lineage differs: {relative}")
        recorded_raw.add(relative)
    if recorded_raw != expected_raw:
        raise ValueError("run receipt raw-artifact inventory differs from the locked matrix")


def _validate_lineage(raw_root: Path, receipt_path: Path) -> Lineage:
    receipt_path = receipt_path.resolve()
    receipt_root = receipt_path.parent
    receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    if receipt.get("tool_revision") != TOOL_REVISION or receipt.get("run_status") != "completed":
        raise ValueError("run receipt is not a completed XY minimal suite")
    if receipt.get("model_saved") is not False or receipt.get("raw_root") != "raw":
        raise ValueError("run receipt does not prove an isolated no-save run")
    if raw_root.resolve() != (receipt_root / "raw").resolve():
        raise ValueError("raw root is not the directory bound by the run receipt")

    source_hashes = _validate_registered_sources(
        receipt_root,
        receipt.get("registered_sources"),
    )
    _validate_raw_artifacts(raw_root, receipt.get("raw_artifacts"))
    log_relative = receipt.get("configuration_log")
    if not isinstance(log_relative, str):
        raise ValueError("run receipt has no configuration log")
    records = _parse_configuration_log((receipt_root / log_relative).resolve())
    compiled_hash = str(receipt.get("compiled_class_sha256", "")).lower()
    if len(compiled_hash) != 64:
        raise ValueError("run receipt has no compiled-class hash")
    version = str(receipt.get("raw_comsol_version", ""))
    if not version.startswith("COMSOL "):
        raise ValueError("run receipt has no raw COMSOL version")
    return Lineage(
        receipt_sha256=_sha256(receipt_path),
        comsol_version=version,
        compiled_class_sha256=compiled_hash,
        source_hashes=source_hashes,
        configuration_record_count=len(records),
    )


def _read_comsol_wide(path: Path) -> tuple[Trajectory, dict[str, str]]:
    metadata: dict[str, str] = {}
    rows: list[list[str]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        for raw in stream:
            if raw.startswith("%"):
                parsed = next(csv.reader([raw[1:].lstrip()]))
                if len(parsed) >= 2:
                    metadata[parsed[0].strip()] = parsed[1].strip()
                continue
            rows.extend(csv.reader([raw]))
    if len(rows) != 1:
        raise ValueError(f"expected one COMSOL particle row in {path}, got {len(rows)}")
    values = np.asarray([float(value) for value in rows[0]], dtype=np.float64)
    if values.size != OUTPUT_TIMES_S.size * 8:
        raise ValueError(f"expected {OUTPUT_TIMES_S.size * 8} values in {path}")
    frame = values.reshape(OUTPUT_TIMES_S.size, 8)
    if not np.all(np.isfinite(frame)):
        raise ValueError(f"non-finite COMSOL state in {path}")
    if not np.all(frame[:, 0] == 1.0):
        raise ValueError(f"unexpected COMSOL particle identity in {path}")
    np.testing.assert_allclose(frame[:, 1], OUTPUT_TIMES_S, rtol=0.0, atol=5.0e-15)
    return (
        Trajectory(
            time_s=frame[:, 1],
            position_m=frame[:, 2:4],
            velocity_m_s=frame[:, 4:6],
            status=frame[:, 6:8].astype(np.int64),
        ),
        metadata,
    )


def _geometry() -> GeometryData:
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype="<f8")
    boundary = BoundaryData(
        line2=np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype="<i8"),
        boundary_id=np.asarray([10, 10, 10, 10], dtype="<i4"),
        group_id=np.asarray([0, 0, 0, 0], dtype="<i4"),
        material_id=np.zeros(4, dtype="<i4"),
        owner_cell_type=np.full(4, 2, dtype="<u1"),
        owner_cell_local_index=np.zeros(4, dtype="<i8"),
        orientation=np.ones(4, dtype="<i1"),
    )
    return GeometryData(
        nodes_m=nodes,
        boundary=boundary,
        group_names=("wall",),
        quad4=np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        quad4_domain_id=np.asarray([0], dtype="<i4"),
    )


def _initial_state(scenario: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    if scenario == "surface_departure":
        return np.asarray([0.4, 0.0]), np.asarray([0.1, 0.2]), 1.0e-12, 0.0
    if scenario in {"specular", "stick", "hold"}:
        return np.asarray([0.8, 0.4]), np.asarray([0.8, 0.1]), 1.0e-12, 0.0
    if scenario == "linear_drag":
        return np.asarray([0.2, 0.3]), np.asarray([0.5, -0.2]), DRAG_MASS_KG, 0.0
    charge = 1.0 if scenario == "electric" else 0.0
    return np.asarray([0.2, 0.3]), np.asarray([0.3, 0.2]), 1.0e-12, charge


def _source(scenario: str) -> RealizedTableSource | RealizedSurfaceSource:
    position, velocity, mass, charge = _initial_state(scenario)
    common = {
        "name": "particle",
        "particle_id": np.asarray([1], dtype="<i8"),
        "release_time_s": np.asarray([0.0], dtype="<f8"),
        "velocity_m_s": velocity.reshape(1, 2).astype("<f8"),
        "charge_number": np.asarray([charge], dtype="<f8"),
        "mass_kg": np.asarray([mass], dtype="<f8"),
        "drag_diameter_m": np.asarray([DRAG_DIAMETER_M], dtype="<f8"),
        "contact_radius_m": np.asarray([0.0], dtype="<f8"),
        "electrostatic_radius_m": np.asarray([0.5 * DRAG_DIAMETER_M], dtype="<f8"),
        "displaced_volume_m3": np.asarray([0.0], dtype="<f8"),
        "model_weight": np.asarray([1.0], dtype="<f8"),
        "material_id": np.asarray([0], dtype="<i4"),
    }
    if scenario == "surface_departure":
        return RealizedSurfaceSource(
            **common,
            facet_id=np.asarray([0], dtype="<i8"),
            facet_parameter=np.asarray([0.4], dtype="<f8"),
        )
    return RealizedTableSource(**common, position_m=position.reshape(1, 2).astype("<f8"))


def _field(
    name: str,
    value: tuple[float, ...],
    components: tuple[str, ...],
    unit: str,
) -> FieldData:
    row = np.asarray(value, dtype="<f8").reshape(1, len(components))
    return FieldData(
        name=name,
        layout="uniform",
        association="node",
        components=components,
        stored_basis="scalar" if len(components) == 1 else "cartesian_xy",
        values=np.repeat(row, 4, axis=0),
        unit=unit,
    )


def _physics_and_fields(scenario: str) -> tuple[dict[str, object], tuple[FieldData, ...]]:
    physics: dict[str, object] = {"charge": {"model": "fixed"}}
    if scenario == "electric":
        physics["electric"] = {
            "model": "coulomb",
            "revision": "electric_coulomb_v1",
            "electric_field": "electric_field",
        }
        return physics, (
            _field(
                "electric_field",
                (ELECTRIC_FIELD_X_V_M, 0.0),
                ("x", "y"),
                "V/m",
            ),
        )
    if scenario == "linear_drag":
        knudsen_radius = 2.0 * DRAG_MEAN_FREE_PATH_M / DRAG_DIAMETER_M
        slip = 1.0 + knudsen_radius * (1.142 + 0.558 * math.exp(-0.999 / knudsen_radius))
        viscosity = DRAG_RATE_S_INV * slip * DRAG_MASS_KG / (3.0 * math.pi * DRAG_DIAMETER_M)
        physics["drag"] = {
            "model": "stokes_cunningham",
            "revision": "stokes_cunningham_allen_raabe_air_v1",
            "gas_velocity_field": "gas_velocity",
            "gas_density_field": "gas_density",
            "gas_dynamic_viscosity_field": "gas_dynamic_viscosity",
            "gas_mean_free_path_field": "gas_mean_free_path",
            "applicability": "error",
        }
        return physics, (
            _field("gas_velocity", tuple(DRAG_GAS_VELOCITY_M_S), ("x", "y"), "m/s"),
            _field("gas_density", (1.0e-12,), ("value",), "kg/m^3"),
            _field("gas_dynamic_viscosity", (viscosity,), ("value",), "Pa*s"),
            _field(
                "gas_mean_free_path",
                (DRAG_MEAN_FREE_PATH_M,),
                ("value",),
                "m",
            ),
        )
    return physics, ()


def _candidate_case(directory: Path, scenario: str, step_s: float) -> Path:
    directory.mkdir(parents=True, exist_ok=False)
    physics, fields = _physics_and_fields(scenario)
    layouts = (
        (
            RegularLayout(
                "uniform",
                np.asarray([0.0, 1.0], dtype="<f8"),
                np.asarray([0.0, 1.0], dtype="<f8"),
                np.asarray([[1]], dtype="<u1"),
            ),
        )
        if fields
        else ()
    )
    provenance = {
        "producer": "tools.vv.comsol.evaluate_xy_minimal_suite",
        "producer_version": TOOL_REVISION,
        "source_sha256": "sha256:" + hashlib.sha256(f"analytic-{scenario}-v1".encode()).hexdigest(),
        "field_semantics_revision": "uniform_analytic_fields_v1" if fields else "no_fields_v1",
        "producer_metadata": {"scenario": scenario, "step_s": step_s},
    }
    data = DataBundle(
        coordinate_system="cartesian_xy",
        provenance_json=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
        geometry=_geometry(),
        layouts=layouts,
        fields=fields,
        sources=(_source(scenario),),
    )
    data_path = directory / "case.h5"
    info = write(data_path, data)
    law = scenario if scenario in {"specular", "stick", "hold"} else "specular"
    source_type: Literal["table", "surface"] = (
        "surface" if scenario == "surface_departure" else "table"
    )
    document = {
        "format_version": 3,
        "case": {
            "name": f"xy_minimal_{scenario}_{step_s:g}",
            "data_path": data_path.name,
            "expected_content_hash": info.content_hash,
        },
        "motion": {"mode": "cartesian_xy"},
        "time": {"start_s": 0.0, "end_s": 0.5, "dt_s": step_s},
        "solver": {
            "integrator": "rk4_fixed",
            "backend": "cpu",
            "seed": 20261009,
            "event": {
                "geometry_rtol": 1.0e-12,
                "roundoff_ulps": 64,
                "max_refinements": 48,
                "max_interactions_per_step": 8,
                "corner_policy": "priority_then_combined_normal_v1",
            },
        },
        "resources": {"memory_limit_mb": 128},
        "physics": physics,
        "sources": [{"name": "release", "type": source_type, "table": "particle"}],
        "boundaries": [{"boundary_group": "wall", "priority": 10, "law": law}],
        "output": {
            "trajectories": {
                "selection": "all",
                "schedule": {"explicit_times_s": OUTPUT_TIMES_S.tolist()},
            },
            "probes": None,
        },
    }
    case_path = directory / "case.yaml"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _candidate_trajectory(result_path: Path) -> tuple[Trajectory, dict[str, object]]:
    result = open_result(result_path)
    frames = list(result.iter_frames())
    if len(frames) != OUTPUT_TIMES_S.size:
        raise ValueError(f"candidate emitted {len(frames)} rather than 11 frames")
    if any(frame.particle_id.tolist() != [1] for frame in frames):
        raise ValueError("candidate particle identity differs")
    trajectory = Trajectory(
        time_s=np.asarray([frame.time_s for frame in frames], dtype=np.float64),
        position_m=np.asarray([frame.position_m[0] for frame in frames], dtype=np.float64),
        velocity_m_s=np.asarray([frame.velocity_m_s[0] for frame in frames], dtype=np.float64),
        status=np.asarray([int(frame.lifecycle[0]) for frame in frames], dtype=np.int64),
    )
    np.testing.assert_allclose(trajectory.time_s, OUTPUT_TIMES_S, rtol=0.0, atol=5.0e-15)
    events = result.read_boundary_events()
    event_rows: list[dict[str, object]] = [
        {
            "time_s": float(events.time_s[index]),
            "particle_id": int(events.particle_id[index]),
            "position_m": events.position_m[index].tolist(),
            "normal": events.normal[index].tolist(),
            "velocity_pre_m_s": events.velocity_pre_m_s[index].tolist(),
            "velocity_post_m_s": events.velocity_post_m_s[index].tolist(),
            "law_id": str(events.law_id[index]),
            "outcome": str(events.outcome[index]),
        }
        for index in range(events.particle_id.size)
    ]
    receipt: dict[str, object] = {
        "manifest": dict(result.manifest),
        "boundary_events": event_rows,
    }
    return trajectory, receipt


def _analytic(scenario: str) -> Trajectory:
    time = OUTPUT_TIMES_S.copy()
    position0, velocity0, mass, _charge = _initial_state(scenario)
    position = position0 + time[:, None] * velocity0
    velocity = np.repeat(velocity0.reshape(1, 2), time.size, axis=0)
    status = np.ones(time.size, dtype=np.int64)
    if scenario == "electric":
        acceleration = np.asarray([ELEMENTARY_CHARGE_C * ELECTRIC_FIELD_X_V_M / mass, 0.0])
        position = position + 0.5 * time[:, None] ** 2 * acceleration
        velocity = velocity + time[:, None] * acceleration
    elif scenario == "linear_drag":
        decay = np.exp(-DRAG_RATE_S_INV * time)
        relative = velocity0 - DRAG_GAS_VELOCITY_M_S
        position = (
            position0
            + time[:, None] * DRAG_GAS_VELOCITY_M_S
            + (1.0 - decay)[:, None] * relative / DRAG_RATE_S_INV
        )
        velocity = DRAG_GAS_VELOCITY_M_S + decay[:, None] * relative
    elif scenario in {"specular", "stick", "hold"}:
        after = time >= EVENT_TIME_S
        if scenario == "specular":
            position[after, 0] = 1.0 - velocity0[0] * (time[after] - EVENT_TIME_S)
            velocity[after, 0] = -velocity0[0]
        else:
            position[after] = np.asarray([1.0, position0[1] + velocity0[1] * EVENT_TIME_S])
            status[after] = 2 if scenario == "stick" else 5
            if scenario == "stick":
                velocity[after] = 0.0
    return Trajectory(time, position, velocity, status)


def _comparison_mask(scenario: str) -> np.ndarray:
    mask = np.ones(OUTPUT_TIMES_S.size, dtype=bool)
    if scenario in {"specular", "stick", "hold"}:
        # COMSOL exports the pre-event velocity on an event-coincident output,
        # while the candidate uses right-continuous post-event state.
        mask &= np.abs(OUTPUT_TIMES_S - EVENT_TIME_S) > 1.0e-14
    return mask


def _maximum_vector_error(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    return float(np.max(np.linalg.norm(left[mask] - right[mask], axis=1)))


def _candidate_event_gate(scenario: str, events: list[object]) -> bool:
    if len(events) != 1 or not isinstance(events[0], dict):
        return False
    event = events[0]
    expected_outcome = {"specular": "reflected", "stick": "stuck", "hold": "held"}[scenario]
    expected_post = {
        "specular": [-0.8, 0.1],
        "stick": [0.0, 0.0],
        "hold": [0.8, 0.1],
    }[scenario]
    return bool(
        event["law_id"] == scenario
        and event["outcome"] == expected_outcome
        and abs(float(event["time_s"]) - EVENT_TIME_S) <= 2.0e-12
        and np.linalg.norm(np.asarray(event["position_m"]) - [1.0, 0.425]) <= 2.0e-12
        and np.linalg.norm(np.asarray(event["normal"]) - [1.0, 0.0]) <= 2.0e-12
        and np.linalg.norm(np.asarray(event["velocity_post_m_s"]) - expected_post) <= 2.0e-12
    )


def _terminal_tail_gate(
    scenario: str,
    comsol: Trajectory,
    candidate: Trajectory,
) -> bool:
    tail = OUTPUT_TIMES_S > EVENT_TIME_S
    if scenario == "specular":
        comsol_ok = bool(np.all(comsol.velocity_m_s[tail, 0] < 0.0))
        candidate_ok = bool(np.all(candidate.status[tail] == 1))
    elif scenario == "stick":
        comsol_ok = bool(np.all(comsol.velocity_m_s[tail] == 0.0))
        candidate_ok = bool(np.all(candidate.status[tail] == 2))
    else:
        comsol_ok = bool(np.all(comsol.position_m[tail] == comsol.position_m[tail][0]))
        candidate_ok = bool(np.all(candidate.status[tail] == 5))
    return comsol_ok and candidate_ok


def _semantic_gate(
    scenario: str,
    comsol: Trajectory,
    candidate: Trajectory,
    receipt: dict[str, object],
) -> bool:
    events = receipt["boundary_events"]
    if not isinstance(events, list):
        return False
    if scenario in {"ballistic", "electric", "linear_drag"}:
        return not events
    if scenario == "surface_departure":
        return bool(
            not events and candidate.position_m[1, 1] > 0.0 and comsol.position_m[1, 1] > 0.0
        )
    return _candidate_event_gate(scenario, events) and _terminal_tail_gate(
        scenario,
        comsol,
        candidate,
    )


def _metric(
    scenario: str,
    step_s: float,
    comsol: Trajectory,
    candidate: Trajectory,
    receipt: dict[str, object],
) -> ScenarioMetrics:
    analytic = _analytic(scenario)
    mask = _comparison_mask(scenario)
    cross_position = _maximum_vector_error(comsol.position_m, candidate.position_m, mask)
    cross_velocity = _maximum_vector_error(comsol.velocity_m_s, candidate.velocity_m_s, mask)
    comsol_position = _maximum_vector_error(comsol.position_m, analytic.position_m, mask)
    comsol_velocity = _maximum_vector_error(comsol.velocity_m_s, analytic.velocity_m_s, mask)
    candidate_position = _maximum_vector_error(candidate.position_m, analytic.position_m, mask)
    candidate_velocity = _maximum_vector_error(candidate.velocity_m_s, analytic.velocity_m_s, mask)
    tolerance = 2.0e-7 if scenario == "linear_drag" else 2.0e-10
    numeric_gate = (
        max(
            cross_position,
            cross_velocity,
            comsol_position,
            comsol_velocity,
            candidate_position,
            candidate_velocity,
        )
        <= tolerance
    )
    events = receipt["boundary_events"]
    return ScenarioMetrics(
        scenario=scenario,
        step_s=step_s,
        maximum_cross_position_error_m=cross_position,
        maximum_cross_velocity_error_m_s=cross_velocity,
        maximum_comsol_analytic_position_error_m=comsol_position,
        maximum_comsol_analytic_velocity_error_m_s=comsol_velocity,
        maximum_candidate_analytic_position_error_m=candidate_position,
        maximum_candidate_analytic_velocity_error_m_s=candidate_velocity,
        compared_frame_count=int(np.count_nonzero(mask)),
        candidate_boundary_event_count=len(events) if isinstance(events, list) else -1,
        semantic_gate=_semantic_gate(scenario, comsol, candidate, receipt),
        numeric_gate=numeric_gate,
    )


def _write_trajectory_rows(
    writer: csv.DictWriter,
    scenario: str,
    step_s: float,
    solver: str,
    trajectory: Trajectory,
) -> None:
    for index, time_s in enumerate(trajectory.time_s):
        status = trajectory.status[index]
        writer.writerow(
            {
                "scenario": scenario,
                "step_s": f"{step_s:.17g}",
                "solver": solver,
                "time_s": f"{time_s:.17g}",
                "x_m": f"{trajectory.position_m[index, 0]:.17g}",
                "y_m": f"{trajectory.position_m[index, 1]:.17g}",
                "vx_m_s": f"{trajectory.velocity_m_s[index, 0]:.17g}",
                "vy_m_s": f"{trajectory.velocity_m_s[index, 1]:.17g}",
                "status": json.dumps(
                    status.tolist() if isinstance(status, np.ndarray) else int(status)
                ),
            }
        )


def _observed_drag_order(metrics: list[ScenarioMetrics], attribute: str) -> float:
    ordered = sorted(
        (item for item in metrics if item.scenario == "linear_drag"), key=lambda x: -x.step_s
    )
    errors = np.asarray([getattr(item, attribute) for item in ordered], dtype=np.float64)
    if errors.size != 3 or np.any(errors <= np.finfo(np.float64).eps):
        return math.nan
    return float(np.polyfit(np.log(np.asarray(STEPS_S)), np.log(errors), 1)[0])


def _evaluate_matrix(
    raw_root: Path,
    trajectory_path: Path,
    lineage: Lineage,
) -> tuple[
    list[ScenarioMetrics],
    list[dict[str, object]],
    list[dict[str, object]],
    set[str],
]:
    columns = [
        "scenario",
        "step_s",
        "solver",
        "time_s",
        "x_m",
        "y_m",
        "vx_m_s",
        "vy_m_s",
        "status",
    ]
    metrics: list[ScenarioMetrics] = []
    raw_artifacts: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    comsol_versions: set[str] = set()
    with (
        tempfile.TemporaryDirectory(prefix="xy-minimal-candidate-") as temporary,
        trajectory_path.open("x", newline="", encoding="utf-8") as stream,
    ):
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        work_root = Path(temporary)
        for scenario in SCENARIOS:
            for step_s in STEPS_S:
                raw_path = raw_root / scenario / _step_directory(step_s) / "state_raw_wide.csv"
                comsol, metadata = _read_comsol_wide(raw_path)
                version = metadata.get("Version", "unknown")
                if version != lineage.comsol_version:
                    raise ValueError(
                        f"raw COMSOL version {version!r} differs from the bound receipt"
                    )
                comsol_versions.add(version)
                raw_artifacts.append(
                    {
                        "path": str(raw_path),
                        "sha256": _sha256(raw_path),
                        "bytes": raw_path.stat().st_size,
                        "comsol_version": version,
                    }
                )
                run_root = work_root / scenario / _step_directory(step_s)
                case_path = _candidate_case(run_root / "case", scenario, step_s)
                result_path = run_root / "result"
                simulate(load_case(case_path), result_path)
                candidate, receipt = _candidate_trajectory(result_path)
                metrics.append(_metric(scenario, step_s, comsol, candidate, receipt))
                receipts.append(
                    {
                        "scenario": scenario,
                        "step_s": step_s,
                        "candidate_case_sha256": _sha256(case_path),
                        "candidate_data_sha256": _sha256(case_path.with_name("case.h5")),
                        "candidate_manifest": receipt["manifest"],
                        "candidate_boundary_events": receipt["boundary_events"],
                    }
                )
                _write_trajectory_rows(writer, scenario, step_s, "comsol", comsol)
                _write_trajectory_rows(writer, scenario, step_s, "candidate", candidate)
    return metrics, raw_artifacts, receipts, comsol_versions


def _build_summary(
    metrics: list[ScenarioMetrics],
    comsol_versions: set[str],
    lineage: Lineage,
) -> dict[str, object]:
    all_numeric = all(item.numeric_gate for item in metrics)
    all_semantic = all(item.semantic_gate for item in metrics)
    finest_numeric = all(item.numeric_gate for item in metrics if item.step_s == min(STEPS_S))
    return {
        "tool_revision": TOOL_REVISION,
        "scientific_status": "PASS" if all_numeric and all_semantic and finest_numeric else "FAIL",
        "tested_scope": (
            "fresh unsaved COMSOL 6.4 Cartesian XY point-particle microcases: ballistic, "
            "constant electric acceleration, linear relaxation, exact surface-origin departure, "
            "specular, Stick, and Freeze/hold"
        ),
        "judgments": {
            "formulation_equivalence": "PASS",
            "field_equivalence": "PASS",
            "boundary_initial_equivalence": "PASS" if all_semantic else "FAIL",
            "trajectory_equivalence": "PASS" if all_numeric else "FAIL",
        },
        "not_tested": [
            "COMSOL exact boundary-event time, candidate facet normal, and residual because the raw COMSOL export contains states rather than an event table",
            "COMSOL native curved elements",
            "finite-radius contact",
            "Brownian, plasma charging, ion drag, DEP, thermophoresis, and lift",
            "general arbitrary COMSOL models outside these isolated meaning tests",
        ],
        "comsol_versions": sorted(comsol_versions),
        "source_lineage": asdict(lineage),
        "model_saved": False,
        "scenario_count": len(SCENARIOS),
        "step_sizes_s": list(STEPS_S),
        "output_times_s": OUTPUT_TIMES_S.tolist(),
        "event_coincident_velocity_convention": {
            "excluded_time_s": EVENT_TIME_S,
            "reason": "COMSOL raw row is pre-event while candidate frame is post-event",
        },
        "metrics": [asdict(item) for item in metrics],
        "drag_observed_orders": {
            "candidate_position": _observed_drag_order(
                metrics, "maximum_candidate_analytic_position_error_m"
            ),
            "candidate_velocity": _observed_drag_order(
                metrics, "maximum_candidate_analytic_velocity_error_m_s"
            ),
            "comsol_position": _observed_drag_order(
                metrics, "maximum_comsol_analytic_position_error_m"
            ),
            "comsol_velocity": _observed_drag_order(
                metrics, "maximum_comsol_analytic_velocity_error_m_s"
            ),
        },
    }


def evaluate(
    raw_root: Path,
    output_directory: Path,
    *,
    run_receipt: Path,
) -> dict[str, object]:
    raw_root = raw_root.resolve()
    output_directory = output_directory.resolve()
    lineage = _validate_lineage(raw_root, run_receipt)
    output_directory.mkdir(parents=True, exist_ok=False)
    metrics, raw_artifacts, receipts, comsol_versions = _evaluate_matrix(
        raw_root,
        output_directory / "normalized_trajectories.csv",
        lineage,
    )

    metric_fields = list(asdict(metrics[0]))
    with (output_directory / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(asdict(item) for item in metrics)
    summary = _build_summary(metrics, comsol_versions, lineage)
    (output_directory / "comparison_summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_directory / "comparison_manifest.json").write_text(
        json.dumps(
            {"raw_comsol_artifacts": raw_artifacts, "candidate_run_receipts": receipts},
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    readme = f"""# Cartesian XY minimal COMSOL comparison

Status: **{summary["scientific_status"]}**

This package compares seven fresh, unsaved COMSOL microcases with candidate
runs made only through `load_case`, `simulate`, and `open_result`.  Both are
also checked against independent closed-form motion.  The three fixed steps
are {list(STEPS_S)} s; the comparison is not tied to matching private solver
internals.

The event-coincident velocity at {EVENT_TIME_S} s is excluded for the three
wall cases because the COMSOL state export is left-continuous there and the
candidate frame is right-continuous.  Pre-event and post-event rows, terminal
tail semantics, and the candidate event payload remain gated.

The result is scoped to the seven isolated Cartesian meanings above.  In
particular it is not evidence for arbitrary COMSOL models or curved elements.
"""
    (output_directory / "README.md").write_text(readme, encoding="utf-8")
    if summary["scientific_status"] != "PASS":
        raise RuntimeError("Cartesian XY minimal comparison gates failed")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-receipt", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.raw_root, args.output, run_receipt=args.run_receipt),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
