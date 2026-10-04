from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest
from tools.vv.comsol import evaluate_m3c1_common_field as common_field
from tools.vv.comsol import evaluate_m3c1_material_event as evaluation

from chamber_particles import open_result as public_open_result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _times() -> list[float]:
    return [*(index * 1.0e-5 for index in range(46)), 4.58e-4, 4.5875e-4]


def _trajectory_row(
    particle_id: int,
    time_s: float,
    *,
    terminal: bool,
    stuck_velocity_m_s: tuple[float, float] = (0.0, 0.0),
) -> tuple[object, ...]:
    if particle_id == 57:
        position = (0.1, 0.022)
        velocity = stuck_velocity_m_s if terminal else (0.25, -0.5)
        charge = -10.0
    else:
        position = (0.05 + particle_id * 1.0e-6 + 0.01 * time_s, 0.03 - 0.02 * time_s)
        velocity = (0.01, -0.02)
        charge = -5.0
    return (
        particle_id,
        time_s,
        *position,
        *velocity,
        charge,
        "stuck" if terminal else "active",
    )


def _write_trajectories(
    parent: Path,
    extended: Path,
    times: list[float],
    *,
    extra_terminal_particle: int | None = None,
    stuck_velocity_m_s: tuple[float, float] = (0.0, 0.0),
) -> None:
    parent.parent.mkdir(parents=True, exist_ok=True)
    extended.parent.mkdir(parents=True, exist_ok=True)
    with (
        parent.open("w", encoding="utf-8", newline="") as parent_stream,
        extended.open("w", encoding="utf-8", newline="") as extended_stream,
    ):
        parent_writer = csv.writer(parent_stream, lineterminator="\n")
        extended_writer = csv.writer(extended_stream, lineterminator="\n")
        parent_writer.writerow(evaluation.TRAJECTORY_COLUMNS)
        extended_writer.writerow(evaluation.TRAJECTORY_COLUMNS)
        for particle_id in range(1, 288):
            for frame, time_s in enumerate(times):
                terminal = (
                    particle_id == 57 or particle_id == extra_terminal_particle
                ) and frame >= 46
                row = _trajectory_row(
                    particle_id,
                    time_s,
                    terminal=terminal,
                    stuck_velocity_m_s=stuck_velocity_m_s,
                )
                extended_writer.writerow(row)
                if frame < 46:
                    parent_writer.writerow(row)


def _write_event(
    path: Path,
    event_time_s: float,
    *,
    boundary_id: int = 6,
    post_velocity_m_s: tuple[float, float] = (0.0, 0.0),
) -> dict[str, object]:
    row: tuple[object, ...] = (
        57,
        1,
        event_time_s,
        0.1,
        0.022,
        0.0,
        1.0,
        0.25,
        -0.5,
        *post_velocity_m_s,
        -10.0,
        -10.0,
        0,
        boundary_id,
        "stick",
        "stuck",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(evaluation.EVENT_COLUMNS)
        writer.writerow(row)
    return {
        "particle_id": 57,
        "event_time_s": event_time_s,
        "hit_position_m": [0.1, 0.022],
        "velocity_pre_m_s": [0.25, -0.5],
        "velocity_post_m_s": list(post_velocity_m_s),
        "charge_number_pre_e": -10.0,
        "charge_number_post_e": -10.0,
        "primary_facet_id": 0,
        "boundary_id": boundary_id,
        "law": "stick",
        "outcome": "stuck",
    }


def _write_common_reference(candidate_path: Path, reference_path: Path) -> None:
    with (
        candidate_path.open(encoding="utf-8", newline="") as candidate_stream,
        reference_path.open("w", encoding="utf-8", newline="") as reference_stream,
    ):
        reader = csv.DictReader(candidate_stream, strict=True)
        writer = csv.DictWriter(
            reference_stream,
            fieldnames=evaluation.TRAJECTORY_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in reader:
            if float(row["time_s"]) > 0.0:
                row["r_m"] = repr(float(row["r_m"]) + 1.0e-10)
                row["velocity_r_m_per_s"] = repr(float(row["velocity_r_m_per_s"]) + 1.0e-8)
                row["charge_number_e"] = repr(float(row["charge_number_e"]) + 1.0e-7)
            writer.writerow(row)


def _write_locked_common_comparison(
    solver_root: Path,
    candidate_path: Path,
    comparison_path: Path,
    *,
    fabricated_zero_gates: bool,
    stale_trajectory_difference: bool,
) -> None:
    common_config = solver_root / "common_p1_evaluation_config.json"
    common_config.write_bytes(
        (
            Path(common_field.__file__).with_name("cases") / "m3c1_caseA_100nm_common_p1_v1.json"
        ).read_bytes()
    )
    field_config = common_field._load_config(common_config)
    common_reference = solver_root / "common_p1_reference.csv"
    _write_common_reference(candidate_path, common_reference)
    budget_path = solver_root / "locked_budget.json"
    fine_artifacts = {
        "candidate": {
            "path": str(candidate_path.resolve()),
            "sha256": _sha256(candidate_path),
        },
        "reference": {
            "path": str(common_reference.resolve()),
            "sha256": _sha256(common_reference),
        },
    }
    _write_json(
        budget_path,
        {
            "schema_version": 1,
            "tool_revision": common_field.TOOL_REVISION,
            "report_kind": "m3c1_preregistered_common_field_budget",
            "status": "REGISTERED",
            "evaluation_config": {
                "path": str(common_config.resolve()),
                "sha256": _sha256(common_config),
            },
            "scope": field_config.metrics.raw["scope"],
            "policy": {"result_dependent_tolerance_tuning": "PROHIBITED"},
            "fine_trajectory_artifacts": fine_artifacts,
            "initial_state_gate": {"status": "PASS"},
            "acceptance": {
                "absolute_limits": field_config.absolute_limits,
                "relative_l2_limits": field_config.metrics.relative_limits,
            },
            "blockers": [],
        },
    )
    comparison = common_field.compare(
        common_config,
        budget_path,
        candidate_path,
        common_reference,
    )
    if fabricated_zero_gates:
        comparison_gates = comparison["acceptance_gates"]
        assert isinstance(comparison_gates, dict)
        for quantity in ("position", "velocity", "charge"):
            quantity_gates = comparison_gates[quantity]
            assert isinstance(quantity_gates, dict)
            for metric in ("rms", "maximum", "relative_l2"):
                gate = quantity_gates[metric]
                assert isinstance(gate, dict)
                gate["observed"] = 0.0
                gate["pass"] = True
    if stale_trajectory_difference:
        trajectory_difference = comparison["trajectory_difference"]
        assert isinstance(trajectory_difference, dict)
        position_difference = trajectory_difference["position"]
        assert isinstance(position_difference, dict)
        position_difference["rms"] = 0.0
    _write_json(comparison_path, comparison)


def _write_input(path: Path, *, segment_z_m: float = 0.022) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        boundary = handle.create_group("geometry/boundary")
        boundary.create_dataset("external_id", data=[134])
        boundary.create_dataset("boundary_id", data=[6])
        boundary.create_dataset("group_id", data=[0])
        boundary.create_dataset("line2", data=[[0, 1]])
        handle.create_dataset("geometry/nodes_m", data=[[0.09, segment_z_m], [0.11, segment_z_m]])
        groups = handle.create_group("geometry/groups")
        groups.create_dataset("names", data=["wafer"], dtype=h5py.string_dtype("utf-8"))


def _write_reference_state(
    path: Path,
    times: list[float],
    event_time_s: float,
    *,
    terminal_start_frame: int = 46,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for particle_id in range(1, 288):
            values: list[float] = []
            for frame, time_s in enumerate(times):
                terminal = particle_id == 57 and frame >= terminal_start_frame
                if particle_id == 57:
                    r_m, z_m = 0.1, 0.022
                    charge = -10.003
                    final_status = 3.0
                else:
                    r_m, z_m = 0.05 + particle_id * 1.0e-6, 0.03
                    charge = -5.0
                    final_status = 1.0
                values.extend(
                    (
                        float(particle_id),
                        time_s,
                        r_m,
                        z_m,
                        0.25,
                        -0.5,
                        charge,
                        3.0 if terminal else 1.0,
                        final_status,
                        event_time_s if particle_id == 57 else 0.0,
                        0.0,
                        1.0e-18,
                        0.25,
                        -0.5,
                        charge,
                    )
                )
            writer.writerow(values)


def _write_ledger(path: Path, root: Path, artifacts: list[Path]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "sha256", "bytes"), lineterminator="\n")
        writer.writeheader()
        for artifact in artifacts:
            writer.writerow(
                {
                    "path": artifact.relative_to(root).as_posix(),
                    "sha256": _sha256(artifact),
                    "bytes": artifact.stat().st_size,
                }
            )


class _FakeResult:
    def __init__(
        self,
        manifest: dict[str, object],
        times: list[float],
        event_time_s: float,
        boundary_id: int,
        extra_terminal_particle: int | None,
        position_offset_m: float,
        event_post_velocity_m_s: tuple[float, float],
        stuck_velocity_m_s: tuple[float, float],
    ) -> None:
        self.manifest = manifest
        self._frames: list[SimpleNamespace] = []
        particle_ids = np.arange(1, 288, dtype=np.int64)
        for frame_index, time_s in enumerate(times):
            rows = [
                _trajectory_row(
                    particle_id,
                    time_s,
                    terminal=(particle_id == 57 or particle_id == extra_terminal_particle)
                    and frame_index >= 46,
                    stuck_velocity_m_s=stuck_velocity_m_s,
                )
                for particle_id in range(1, 288)
            ]
            self._frames.append(
                SimpleNamespace(
                    time_s=time_s,
                    particle_id=particle_ids,
                    position_m=(
                        np.asarray([row[2:4] for row in rows], dtype=np.float64) + position_offset_m
                    ),
                    velocity_m_s=np.asarray([row[4:6] for row in rows], dtype=np.float64),
                    charge_number=np.asarray([row[6] for row in rows], dtype=np.float64),
                    lifecycle=np.asarray(
                        [2 if row[7] == "stuck" else 1 for row in rows], dtype=np.uint8
                    ),
                )
            )
        self._events = SimpleNamespace(
            particle_id=np.asarray([57], dtype=np.int64),
            event_ordinal=np.asarray([1], dtype=np.uint32),
            time_s=np.asarray([event_time_s], dtype=np.float64),
            position_m=np.asarray([[0.1, 0.022]], dtype=np.float64),
            normal=np.asarray([[0.0, 1.0]], dtype=np.float64),
            velocity_pre_m_s=np.asarray([[0.25, -0.5]], dtype=np.float64),
            velocity_post_m_s=np.asarray([event_post_velocity_m_s], dtype=np.float64),
            charge_number_pre=np.asarray([-10.0], dtype=np.float64),
            charge_number_post=np.asarray([-10.0], dtype=np.float64),
            primary_facet_id=np.asarray([0], dtype=np.int64),
            boundary_id=np.asarray([boundary_id], dtype=np.int32),
            law_id=np.asarray(["stick"]),
            outcome=np.asarray(["stuck"]),
        )

    def iter_frames(self) -> object:
        return iter(self._frames)

    def read_boundary_events(self) -> SimpleNamespace:
        return self._events

    def read_failure_events(self) -> SimpleNamespace:
        return SimpleNamespace(particle_id=np.empty(0, dtype=np.int64))


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate_event_time_s: float = 4.575e-4,
    candidate_boundary_id: int = 6,
    reference_terminal_start_frame: int = 46,
    segment_z_m: float = 0.022,
    extra_terminal_particle: int | None = None,
    durable_position_offset_m: float = 0.0,
    event_post_velocity_m_s: tuple[float, float] = (0.0, 0.0),
    stuck_velocity_m_s: tuple[float, float] = (0.0, 0.0),
    fabricated_zero_gates: bool = False,
    stale_trajectory_difference: bool = False,
) -> dict[str, Path]:
    solver_root = tmp_path / "solver"
    parent_root = solver_root / "parent"
    candidate_root = solver_root / "candidate"
    reference_root = solver_root / "reference"
    times = _times()
    input_path = parent_root / "candidate_input.h5"
    parent_trajectory = parent_root / "parent_trajectory.csv"
    candidate_trajectory = candidate_root / "candidate_trajectory.csv"
    candidate_pre_event = candidate_root / evaluation.PRE_EVENT_TRAJECTORY_FILENAME
    _write_input(input_path, segment_z_m=segment_z_m)
    _write_trajectories(
        parent_trajectory,
        candidate_trajectory,
        times,
        extra_terminal_particle=extra_terminal_particle,
        stuck_velocity_m_s=stuck_velocity_m_s,
    )
    candidate_pre_event.write_bytes(parent_trajectory.read_bytes())
    parent_case = parent_root / "parent_case.yaml"
    parent_manifest = parent_root / "parent_result" / "run.json"
    parent_run_report = parent_root / "parent_run_report.json"
    pre_event_receipt = solver_root / "pre_event_reference" / "common_p1_run_receipt.json"
    pre_event_comparison = solver_root / "pre_event_comparison.json"
    parent_case.write_text("parent case\n", encoding="utf-8")
    _write_json(parent_manifest, {"status": "complete"})
    _write_json(parent_run_report, {"status": "COMPLETE"})
    _write_json(pre_event_receipt, {"status": "COMPLETE"})
    _write_locked_common_comparison(
        solver_root,
        parent_trajectory,
        pre_event_comparison,
        fabricated_zero_gates=fabricated_zero_gates,
        stale_trajectory_difference=stale_trajectory_difference,
    )
    config: dict[str, Any] = {
        "schema_version": 1,
        "evaluation_id": evaluation.EVALUATION_ID,
        "evaluation_revision": 1,
        "classification": "external_same_field_first_material_event_anchor",
        "expected_comsol_version": "6.4.0.429",
        "source_model": {"sha256": "a" * 64},
        "case": {
            "workflow": "caseA",
            "diameter_m": 1.0e-7,
            "particle_count": 287,
            "pre_event_end_s": 4.5e-4,
            "event_window_end_s": times[-1],
            "fixed_rk4_step_s": 1.5625e-7,
            "output_times_s": times,
        },
        "candidate": {
            "parent_root_relative_path": "parent",
            "input_filename": input_path.name,
            "input_sha256": _sha256(input_path),
            "input_content_hash": "sha256:" + "b" * 64,
            "parent_trajectory_filename": parent_trajectory.name,
            "parent_trajectory_sha256": _sha256(parent_trajectory),
            "parent_case_filename": parent_case.name,
            "parent_case_sha256": _sha256(parent_case),
            "parent_result_manifest_relative_path": parent_manifest.relative_to(
                parent_root
            ).as_posix(),
            "parent_result_manifest_sha256": _sha256(parent_manifest),
            "parent_run_report_filename": parent_run_report.name,
            "parent_run_report_sha256": _sha256(parent_run_report),
        },
        "reference": {
            "pre_event_root_relative_path": "pre_event_reference",
            "pre_event_run_receipt_sha256": _sha256(pre_event_receipt),
            "pre_event_comparison_relative_path": pre_event_comparison.relative_to(
                solver_root
            ).as_posix(),
            "pre_event_comparison_sha256": _sha256(pre_event_comparison),
        },
        "expected_first_event": {
            "particle_id": 57,
            "candidate_boundary_id": 6,
            "candidate_external_id": 134,
            "boundary_group": "wafer",
            "comsol_status_code": 3,
            "candidate_lifecycle": "stuck",
            "candidate_law": "stick",
            "wafer_z_m": 0.022,
        },
        "acceptance": {
            "event_time_absolute_s": 1.0e-9,
            "hit_position_norm_m": 2.0e-9,
            "terminal_position_hold_m": 2.0e-12,
            "terminal_charge_cross_absolute_e": 0.0045,
            "terminal_charge_hold_roundoff_multiplier": 4096.0,
        },
        "claim_policy": evaluation.LOCKED_CLAIM_POLICY,
    }
    config_path = solver_root / "config.json"
    _write_json(config_path, config)
    config_hash = _sha256(config_path)
    monkeypatch.setattr(evaluation, "LOCKED_CONFIG_SHA256", config_hash)
    candidate_root.mkdir(parents=True, exist_ok=True)
    event_path = candidate_root / "candidate_events.csv"
    first_event = _write_event(
        event_path,
        candidate_event_time_s,
        boundary_id=candidate_boundary_id,
        post_velocity_m_s=event_post_velocity_m_s,
    )
    prepare_path = candidate_root / "prepare_report.json"
    case_path = candidate_root / "candidate_material_event.yaml"
    case_path.write_text("case: synthetic\n", encoding="utf-8")
    producer_path = candidate_root / evaluation.CANDIDATE_PRODUCER_FILENAME
    producer_path.write_bytes(
        Path(evaluation.__file__).with_name(evaluation.CANDIDATE_PRODUCER_FILENAME).read_bytes()
    )
    producer = {
        "file": evaluation.CANDIDATE_PRODUCER_FILENAME,
        "sha256": _sha256(producer_path),
    }
    monkeypatch.setattr(evaluation, "LOCKED_CANDIDATE_CASE_SHA256", _sha256(case_path))
    locked_inputs = {
        "case": {"path": str(parent_case.resolve()), "sha256": _sha256(parent_case)},
        "comparison": {
            "path": str(pre_event_comparison.resolve()),
            "sha256": _sha256(pre_event_comparison),
        },
        "input": {"path": str(input_path.resolve()), "sha256": _sha256(input_path)},
        "manifest": {
            "path": str(parent_manifest.resolve()),
            "sha256": _sha256(parent_manifest),
        },
        "reference_receipt": {
            "path": str(pre_event_receipt.resolve()),
            "sha256": _sha256(pre_event_receipt),
        },
        "run_report": {
            "path": str(parent_run_report.resolve()),
            "sha256": _sha256(parent_run_report),
        },
        "trajectory": {
            "path": str(parent_trajectory.resolve()),
            "sha256": _sha256(parent_trajectory),
        },
    }
    _write_json(
        prepare_path,
        {
            "status": "PREPARED",
            "tool_revision": evaluation.CANDIDATE_TOOL_REVISION,
            "configuration_sha256": config_hash,
            "configuration_sha256_before": config_hash,
            "configuration_sha256_after": config_hash,
            "producer_source": producer,
            "case_sha256": _sha256(case_path),
            "scope": {
                "particles": 287,
                "frames": 48,
                "time_window_s": [0.0, times[-1]],
                "dt_s": 1.5625e-7,
            },
            "claim_policy": evaluation.LOCKED_CLAIM_POLICY,
            "locked_inputs": locked_inputs,
        },
    )
    manifest_path = candidate_root / "result_material_event" / "run.json"
    _write_json(
        manifest_path,
        {
            "status": "complete",
            "case_file_hash": f"sha256:{_sha256(case_path)}",
            "data_content_hash": config["candidate"]["input_content_hash"],
            "engine_algorithm_revision": "particle_engine_v31",
            "boundary_algorithm_revision": "point_wall_laws_v4",
            "event_algorithm_revision": "line_quadratic_rk4_axis_first_hit_v14",
            "physics_runtime_revision": "inertial_langevin_compiled_physics_runtime_v17",
            "motion_mode": "axisymmetric_rz_meridional",
            "brownian_rng_revision": None,
            "time": {
                "start_s": 0.0,
                "end_s": times[-1],
                "dt_s": 1.5625e-7,
            },
            "resolved": {"backend": "cpu", "integrator": "rk4_fixed"},
            "counts": {
                "failure_events": 0,
                "frame_rows": 287 * 48,
                "boundary_events": 1,
                "particles": 287,
                "frames": 48,
                "release_events": 287,
            },
            "lifecycle_counts": {
                "active": 286,
                "pending": 0,
                "stuck": 1,
                "escaped": 0,
                "failed": 0,
            },
        },
    )
    _write_json(
        candidate_root / "candidate_run_report.json",
        {
            "status": "COMPLETE",
            "tool_revision": evaluation.CANDIDATE_TOOL_REVISION,
            "configuration_sha256": config_hash,
            "configuration_sha256_before": config_hash,
            "configuration_sha256_after": config_hash,
            "producer_source": producer,
            "prepare_report_sha256": _sha256(prepare_path),
            "case_sha256": _sha256(case_path),
            "result_manifest_sha256": _sha256(manifest_path),
            "result_manifest_status": "complete",
            "trajectory": candidate_trajectory.name,
            "trajectory_sha256": _sha256(candidate_trajectory),
            "trajectory_rows": 287 * 48,
            "pre_event_trajectory": candidate_pre_event.name,
            "pre_event_trajectory_sha256": _sha256(candidate_pre_event),
            "pre_event_trajectory_rows": 287 * 46,
            "events": event_path.name,
            "events_sha256": _sha256(event_path),
            "event_rows": 1,
            "first_event": first_event,
            "failure_event_count": 0,
            "pre_event_lineage": {
                "historical_parent_trajectory_sha256": _sha256(parent_trajectory),
                "bitwise_parent_equality_required": False,
                "acceptance_owner": "evaluate_m3c1_common_field.py",
            },
            "claim_policy": evaluation.LOCKED_CLAIM_POLICY,
        },
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fake_result = _FakeResult(
        manifest,
        times,
        candidate_event_time_s,
        candidate_boundary_id,
        extra_terminal_particle,
        durable_position_offset_m,
        event_post_velocity_m_s,
        stuck_velocity_m_s,
    )
    monkeypatch.setattr(evaluation, "open_result", lambda path: fake_result)
    state_path = reference_root / "dt_0p15625us" / "state_raw_wide.csv"
    _write_reference_state(
        state_path,
        times,
        4.575e-4,
        terminal_start_frame=reference_terminal_start_frame,
    )
    raw_records: dict[str, dict[str, object]] = {}
    for name, expressions in (
        ("state_raw_wide.csv", 48 * 15),
        ("force_raw_wide.csv", 48 * 20),
        ("primitive_raw_wide.csv", 48 * 24),
    ):
        raw_path = state_path.parent / name
        if name != "state_raw_wide.csv":
            raw_path.write_text("synthetic raw table\n", encoding="utf-8")
        raw_records[name] = {
            "relative_path": raw_path.relative_to(reference_root).as_posix(),
            "sha256": _sha256(raw_path),
            "size_bytes": raw_path.stat().st_size,
            "nodes": 287,
            "expressions": expressions,
            "data_rows": 287,
        }
    validation_path = reference_root / "material_event_raw_validation.json"
    _write_json(
        validation_path,
        {
            "schema_version": 1,
            "tool_revision": evaluation.RAW_VALIDATION_REVISION,
            "status": "PASS",
            "config_sha256": config_hash,
            "raw_tables": raw_records,
        },
    )
    component_files = [f"field_{index:02d}.txt" for index in range(22)]
    release_files = [f"release_{index}.txt" for index in range(3)]
    probe_file = "release_probes.csv"
    prepared_names = [*component_files, *release_files, probe_file]
    prepared_artifacts: dict[str, dict[str, object]] = {}
    for index, name in enumerate(prepared_names):
        artifact = reference_root / name
        artifact.write_text(f"prepared {index}\n", encoding="utf-8")
        prepared_artifacts[artifact.name] = {
            "sha256": _sha256(artifact),
            "size_bytes": artifact.stat().st_size,
        }
    table_receipt_path = reference_root / "common_p1_table_receipt.json"
    _write_json(
        table_receipt_path,
        {
            "schema_version": 1,
            "tool_revision": "m3c1_full_physics_common_p1_tables_v1",
            "classification": "external_vv_full_physics_exact_p1_common_field_input",
            "candidate": {
                "file_sha256": config["candidate"]["input_sha256"],
                "content_hash": config["candidate"]["input_content_hash"],
            },
            "component_count": 22,
            "components": [{"file": name} for name in component_files],
            "release": {
                "functions": [{"file": name} for name in release_files],
                "probe_file": probe_file,
            },
            "artifacts": prepared_artifacts,
        },
    )
    prepared_rows = [
        {"path": name, **record} for name, record in sorted(prepared_artifacts.items())
    ]
    prepared_validation = {
        "schema_version": 1,
        "status": "PASS",
        "criterion": "synthetic exact receipt validation",
        "receipt_sha256": _sha256(table_receipt_path),
        "pre_comsol": {"status": "PASS", "artifact_count": 26},
        "post_comsol": {"status": "PASS", "artifact_count": 26},
        "artifacts": prepared_rows,
    }
    prepared_validation_path = reference_root / "prepared_table_validation.json"
    _write_json(prepared_validation_path, prepared_validation)
    staged_tools = {
        "m3c1_common_p1_material_event_v1.json": config_path.read_bytes(),
        "RunM3C1CaseA100CommonP1.java": b"shared java\n",
        "RunM3C1CaseA100CommonP1MaterialEvent.java": b"entry java\n",
        "prepare_m3c1_common_p1_tables.py": b"preparer\n",
        "validate_m3c1_common_p1_material_event_run.py": b"validator\n",
    }
    for name, payload in staged_tools.items():
        (reference_root / name).write_bytes(payload)
    provenance_path = reference_root / "provenance.json"
    _write_json(
        provenance_path,
        {
            "classification": config["classification"],
            "evaluation_id": evaluation.EVALUATION_ID,
            "evaluation_revision": 1,
            "config_sha256": config_hash,
            "source_sha256_before": "a" * 64,
            "source_sha256_after": "a" * 64,
            "source_unchanged": True,
            "source_load_mode": "ModelUtil.loadCopy",
            "model_saved": False,
            "isolated_source_copy_sha256": "a" * 64,
            "isolated_source_copy_retained": False,
            "candidate_input_sha256": config["candidate"]["input_sha256"],
            "candidate_content_hash": config["candidate"]["input_content_hash"],
            "table_receipt_sha256": _sha256(table_receipt_path),
            "prepared_table_validation_status": "PASS",
            "run_profile": "material_event",
            "process_count": 1,
            "comsol_version": "COMSOL Multiphysics 6.4.0.429",
            "runner_sha256": _sha256(
                Path(evaluation.__file__).with_name("run_m3c1_common_p1_reference.ps1")
            ),
            "staged_config": "m3c1_common_p1_material_event_v1.json",
            "java_source_sha256": _sha256(reference_root / "RunM3C1CaseA100CommonP1.java"),
            "staged_java": "RunM3C1CaseA100CommonP1.java",
            "material_event_entry_java_sha256": _sha256(
                reference_root / "RunM3C1CaseA100CommonP1MaterialEvent.java"
            ),
            "material_event_entry_java": "RunM3C1CaseA100CommonP1MaterialEvent.java",
            "preparer_source_sha256": _sha256(reference_root / "prepare_m3c1_common_p1_tables.py"),
            "staged_preparer": "prepare_m3c1_common_p1_tables.py",
            "postprocessor_source_sha256": _sha256(
                reference_root / "validate_m3c1_common_p1_material_event_run.py"
            ),
            "staged_postprocessor": "validate_m3c1_common_p1_material_event_run.py",
        },
    )
    ledger_path = reference_root / "artifact_hashes.csv"
    _write_ledger(
        ledger_path,
        reference_root,
        sorted(
            (path for path in reference_root.rglob("*") if path.is_file()),
            key=lambda path: path.as_posix(),
        ),
    )
    receipt_path = reference_root / "common_p1_material_event_run_receipt.json"
    _write_json(
        receipt_path,
        {
            "schema_version": 1,
            "tool_revision": evaluation.REFERENCE_TOOL_REVISION,
            "classification": config["classification"],
            "status": "COMPLETE",
            "config_sha256": config_hash,
            "candidate_input_sha256": config["candidate"]["input_sha256"],
            "source_sha256_before": "a" * 64,
            "source_sha256_after": "a" * 64,
            "comsol_version": "COMSOL Multiphysics 6.4.0.429",
            "scope": {
                "particles": 287,
                "frames": 48,
                "time_window_s": [0.0, times[-1]],
                "fixed_rk4_step_s": 1.5625e-7,
            },
            "raw_validation_sha256": _sha256(validation_path),
            "artifact_hashes_sha256": _sha256(ledger_path),
            "table_receipt_sha256": _sha256(table_receipt_path),
            "prepared_table_validation": prepared_validation,
            "raw_tables": raw_records,
        },
    )
    status_path = reference_root / "run_status.json"
    _write_json(
        status_path,
        {
            "status": "COMPLETE",
            "failure": "",
            "source_copy_retained": False,
            "compiled_class_retained": False,
            "class_status_retained": False,
        },
    )
    monkeypatch.setattr(evaluation, "_solver_root", lambda: solver_root)
    return {
        "config": config_path,
        "candidate": candidate_root,
        "pre_event_comparison": pre_event_comparison,
        "receipt": receipt_path,
        "validation": validation_path,
        "state": state_path,
        "ledger": ledger_path,
        "status": status_path,
    }


def _evaluate(paths: dict[str, Path], output: Path) -> dict[str, object]:
    return evaluation.evaluate(
        paths["config"],
        paths["candidate"],
        paths["pre_event_comparison"],
        paths["receipt"],
        paths["validation"],
        paths["state"],
        paths["ledger"],
        paths["status"],
        output,
    )


def test_evaluates_first_material_event_and_characterizes_velocity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)

    result = _evaluate(paths, tmp_path / "evaluation")

    assert result["status"] == "PASS"
    assert result["gate_summary"] == {"pass": 20, "fail": 0}
    material = result["material_event"]
    assert isinstance(material, dict)
    assert material["status_semantics"] == {
        "event_detection_authority": "current_status_code",
        "final_status_code_role": "characterization_only",
    }
    assert material["velocity_semantics"]["cross_solver_gate"] is False
    prefix = result["pre_event_prefix"]
    assert isinstance(prefix, dict)
    same_field = prefix["same_field_comparison"]
    assert isinstance(same_field, dict)
    assert same_field["status"] == "PASS"
    assert same_field["passed_gates"] == 9
    assert (tmp_path / "evaluation" / "comparison_result.json").is_file()
    assert (tmp_path / "evaluation" / "gates.csv").is_file()


def test_rejects_fabricated_zero_pre_event_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, fabricated_zero_gates=True)

    with pytest.raises(ValueError, match="acceptance gates are not reproducible"):
        _evaluate(paths, tmp_path / "evaluation")


def test_rejects_stale_pre_event_trajectory_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, stale_trajectory_difference=True)

    with pytest.raises(ValueError, match="trajectory_difference is not reproducible"):
        _evaluate(paths, tmp_path / "evaluation")


def test_fails_a_nonzero_stick_event_post_velocity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, event_post_velocity_m_s=(1.0e-12, 0.0))

    result = _evaluate(paths, tmp_path / "evaluation")

    assert result["status"] == "FAIL"
    gates = result["gates"]
    assert isinstance(gates, list)
    failed = [gate["gate"] for gate in gates if gate["status"] == "FAIL"]
    assert failed == ["candidate_stick_event_post_velocity_is_zero"]


def test_fails_a_nonzero_saved_stuck_velocity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, stuck_velocity_m_s=(0.0, -1.0e-12))

    result = _evaluate(paths, tmp_path / "evaluation")

    assert result["status"] == "FAIL"
    gates = result["gates"]
    assert isinstance(gates, list)
    failed = [gate["gate"] for gate in gates if gate["status"] == "FAIL"]
    assert failed == ["candidate_all_saved_stuck_velocities_are_zero"]


def test_reports_failed_event_time_gate_without_tuning_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, candidate_event_time_s=4.57502e-4)

    result = _evaluate(paths, tmp_path / "evaluation")

    assert result["status"] == "FAIL"
    gates = result["gates"]
    assert isinstance(gates, list)
    failed = [gate["gate"] for gate in gates if gate["status"] == "FAIL"]
    assert failed == ["event_time_absolute_difference"]


def test_rejects_reference_state_changed_after_receipting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    with paths["state"].open("a", encoding="utf-8") as stream:
        stream.write("changed\n")

    with pytest.raises(ValueError, match=r"state_raw_wide\.csv differs from its receipt"):
        _evaluate(paths, tmp_path / "evaluation")


def test_does_not_mask_a_wrong_raw_event_boundary_with_canonical_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, candidate_boundary_id=999)

    result = _evaluate(paths, tmp_path / "evaluation")

    gates = result["gates"]
    assert isinstance(gates, list)
    failed = [gate["gate"] for gate in gates if gate["status"] == "FAIL"]
    assert failed == ["candidate_first_event_identity"]


def test_rejects_a_reference_that_is_terminal_from_its_first_saved_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, reference_terminal_start_frame=0)

    result = _evaluate(paths, tmp_path / "evaluation")

    gates = result["gates"]
    assert isinstance(gates, list)
    failed = [gate["gate"] for gate in gates if gate["status"] == "FAIL"]
    assert "reference_exactly_one_nonactive_stuck_particle" in failed
    assert "reference_event_time_bracketed_by_saved_status_transition" in failed


def test_rejects_an_event_that_is_not_on_the_canonical_wafer_segment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, segment_z_m=0.023)

    result = _evaluate(paths, tmp_path / "evaluation")

    gates = result["gates"]
    assert isinstance(gates, list)
    failed = [gate["gate"] for gate in gates if gate["status"] == "FAIL"]
    assert "canonical_event_facet_is_expected_wafer_surface" in failed
    assert "reference_hit_on_canonical_event_facet" in failed
    assert "candidate_hit_on_canonical_event_facet" in failed


def test_rejects_an_unreported_second_terminal_particle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, extra_terminal_particle=58)

    result = _evaluate(paths, tmp_path / "evaluation")

    gates = result["gates"]
    assert isinstance(gates, list)
    failed = [gate["gate"] for gate in gates if gate["status"] == "FAIL"]
    assert failed == ["candidate_exactly_one_nonactive_stuck_particle"]


def test_rejects_an_incomplete_reference_run_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    status["status"] = "INCOMPLETE"
    status["failure"] = "synthetic failure"
    _write_json(paths["status"], status)

    with pytest.raises(ValueError, match="reference run status differs"):
        _evaluate(paths, tmp_path / "evaluation")


def test_rejects_an_artifact_omitted_from_the_final_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    (paths["receipt"].parent / "unindexed.txt").write_text("not receipted\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact ledger coverage differs"):
        _evaluate(paths, tmp_path / "evaluation")


def test_rejects_a_missing_durable_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluation, "open_result", public_open_result)

    with pytest.raises(RuntimeError, match="not complete"):
        _evaluate(paths, tmp_path / "evaluation")


def test_rejects_a_corrupt_durable_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    result_root = paths["candidate"] / "result_material_event"
    (result_root / "_SUCCESS").write_bytes(b"")
    monkeypatch.setattr(evaluation, "open_result", public_open_result)

    with pytest.raises(RuntimeError, match="unsupported result schema"):
        _evaluate(paths, tmp_path / "evaluation")


def test_rejects_a_csv_that_differs_from_the_durable_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch, durable_position_offset_m=1.0e-12)

    with pytest.raises(ValueError, match="trajectory values differ"):
        _evaluate(paths, tmp_path / "evaluation")
