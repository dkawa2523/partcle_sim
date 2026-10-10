"""Run the P18-H force-free ``hold`` boundary candidate.

This is an external V&V utility.  It creates one producer-neutral canonical
R-Z input, runs the public solver API, and records a small auditable projection.
It neither imports COMSOL nor adds a COMSOL-specific path to the solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
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

TOOL_REVISION: Final = "p18h_hold_candidate_v1"
CASE_FILENAME: Final = "candidate_case.yaml"
DATA_FILENAME: Final = "candidate_input.h5"
RESULT_DIRECTORY: Final = "result"
TRAJECTORY_FILENAME: Final = "candidate_trajectory.csv"
EVENT_FILENAME: Final = "candidate_events.csv"
REPORT_FILENAME: Final = "candidate_run_report.json"
PRODUCER_FILENAME: Final = "run_p18h_hold_candidate.py"
CONFIG_FILENAME: Final = "p18h_hold_freeze_v1.json"
_LIFECYCLE: Final = {
    0: "pending",
    1: "active",
    2: "stuck",
    3: "escaped",
    4: "failed",
    5: "held",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _solver_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    config = _mapping(json.loads(path.read_text(encoding="utf-8")), "configuration")
    required = {
        "schema_version",
        "evaluation_id",
        "evaluation_revision",
        "classification",
        "case",
        "reference",
        "expected_candidate",
        "acceptance",
        "claim_policy",
    }
    if set(config) != required:
        raise ValueError(f"configuration keys differ: {sorted(set(config) ^ required)}")
    if config["schema_version"] != 1 or config["evaluation_revision"] != 1:
        raise ValueError("unsupported P18-H configuration revision")
    case = _mapping(config["case"], "case")
    if (
        case.get("coordinate_system") != "axisymmetric_rz"
        or case.get("motion_mode") != "axisymmetric_rz_meridional"
        or case.get("particle_id") != 1
        or case.get("output_frames") != 61
    ):
        raise ValueError("P18-H anchor case identity differs")
    if not math.isclose(float(case.get("fixed_rk4_step_s", 0.0)), 2.5e-6):
        raise ValueError("P18-H anchor must use the locked 2.5 us step")
    if not math.isclose(float(case.get("analytic_event_time_s", 0.0)), 7.3e-5):
        raise ValueError("P18-H analytic event-time anchor differs")
    return config


def _locked_references(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    reference = _mapping(config["reference"], "reference")
    definitions = {
        "compact_manifest": ("compact_manifest_relative_path", "compact_manifest_sha256"),
        "compact_gates": ("compact_gates_relative_path", "compact_gates_sha256"),
        "compact_readme": ("compact_readme_relative_path", "compact_readme_sha256"),
        "state": ("state_relative_path", "state_sha256"),
        "event": ("event_relative_path", "event_sha256"),
        "summary": ("summary_relative_path", "summary_sha256"),
    }
    root = _solver_root()
    locked: dict[str, dict[str, str]] = {}
    for name, (path_key, hash_key) in definitions.items():
        path = (root / str(reference[path_key])).resolve()
        expected_hash = str(reference[hash_key])
        if not path.is_file():
            raise FileNotFoundError(f"locked {name} is missing: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"locked {name} hash differs: {path}")
        locked[name] = {"path": str(path), "sha256": actual_hash}
    manifest = json.loads(Path(locked["compact_manifest"]["path"]).read_text(encoding="utf-8"))
    if (
        manifest.get("evaluation_status") != "COMPLETE"
        or manifest.get("scientific_status") != "PASS"
        or manifest.get("gate_counts", {}).get("fail") != 0
    ):
        raise ValueError("locked M3-C0 boundary reference is not a complete PASS")
    return locked


def _candidate_bundle(config: dict[str, Any], config_hash: str) -> DataBundle:
    case = _mapping(config["case"], "case")
    r_min, r_max = (float(value) for value in case["domain_r_m"])
    z_min, z_max = (float(value) for value in case["domain_z_m"])
    nodes = np.asarray(
        [[r_min, z_min], [r_max, z_min], [r_max, z_max], [r_min, z_max]],
        dtype="<f8",
    )
    boundary = BoundaryData(
        line2=np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype="<i8"),
        boundary_id=np.asarray([10, 37, 10, 10], dtype="<i4"),
        group_id=np.zeros(4, dtype="<i4"),
        material_id=np.zeros(4, dtype="<i4"),
        owner_cell_type=np.full(4, 2, dtype="<u1"),
        owner_cell_local_index=np.zeros(4, dtype="<i8"),
        orientation=np.ones(4, dtype="<i1"),
        external_id=np.asarray([10, 37, 30, 40], dtype="<i8"),
    )
    geometry = GeometryData(
        nodes_m=nodes,
        boundary=boundary,
        group_names=("hold_probe",),
        quad4=np.asarray([[0, 1, 2, 3]], dtype="<i8"),
        quad4_domain_id=np.asarray([0], dtype="<i4"),
    )
    diameter = float(case["particle_diameter_m"])
    source = RealizedTableSource(
        name="particles",
        particle_id=np.asarray([case["particle_id"]], dtype="<i8"),
        release_time_s=np.asarray([0.0], dtype="<f8"),
        position_m=np.asarray([case["source_position_m"]], dtype="<f8"),
        velocity_m_s=np.asarray([case["source_velocity_m_per_s"]], dtype="<f8"),
        charge_number=np.asarray([case["source_charge_number_e"]], dtype="<f8"),
        mass_kg=np.asarray([case["particle_mass_kg"]], dtype="<f8"),
        drag_diameter_m=np.asarray([diameter], dtype="<f8"),
        contact_radius_m=np.asarray([0.0], dtype="<f8"),
        electrostatic_radius_m=np.asarray([0.5 * diameter], dtype="<f8"),
        displaced_volume_m3=np.asarray([math.pi * diameter**3 / 6.0], dtype="<f8"),
        model_weight=np.asarray([1.0], dtype="<f8"),
        material_id=np.asarray([0], dtype="<i4"),
    )
    provenance = json.dumps(
        {
            "producer": "p18h-hold-freeze-vv",
            "producer_version": TOOL_REVISION,
            "source_sha256": f"sha256:{config_hash}",
            "field_semantics_revision": "force_free_v1",
            "producer_metadata": {"evaluation_id": config["evaluation_id"]},
        }
    )
    return DataBundle("axisymmetric_rz", provenance, geometry, sources=(source,))


def _case_document(config: dict[str, Any], content_hash: str) -> dict[str, object]:
    case = _mapping(config["case"], "case")
    times = np.linspace(0.0, float(case["time_end_s"]), int(case["output_frames"])).tolist()
    return {
        "format_version": 3,
        "case": {
            "name": "p18h_hold_freeze_force_free",
            "data_path": DATA_FILENAME,
            "expected_content_hash": content_hash,
        },
        "motion": {"mode": case["motion_mode"]},
        "time": {
            "start_s": 0.0,
            "end_s": float(case["time_end_s"]),
            "dt_s": float(case["fixed_rk4_step_s"]),
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
        "sources": [{"name": "release", "type": "table", "table": "particles"}],
        "boundaries": [{"boundary_group": "hold_probe", "priority": 10, "law": "hold"}],
        "output": {
            "trajectories": {
                "selection": "all",
                "schedule": {"explicit_times_s": times},
            }
        },
    }


def _write_projection(root: Path, result: Any) -> tuple[int, int]:
    trajectory_rows = 0
    with (root / TRAJECTORY_FILENAME).open("x", encoding="utf-8", newline="") as stream:
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
        for frame in result.iter_frames():
            for row, particle_id in enumerate(frame.particle_id):
                writer.writerow(
                    (
                        int(particle_id),
                        frame.time_s,
                        *frame.position_m[row],
                        *frame.velocity_m_s[row],
                        frame.charge_number[row],
                        _LIFECYCLE[int(frame.lifecycle[row])],
                    )
                )
                trajectory_rows += 1

    events = result.read_boundary_events()
    with (root / EVENT_FILENAME).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            (
                "particle_id",
                "event_ordinal",
                "event_time_s",
                "hit_r_m",
                "hit_z_m",
                "pre_velocity_r_m_per_s",
                "pre_velocity_z_m_per_s",
                "post_velocity_r_m_per_s",
                "post_velocity_z_m_per_s",
                "charge_number_pre_e",
                "charge_number_post_e",
                "boundary_id",
                "law",
                "outcome",
            )
        )
        for row, particle_id in enumerate(events.particle_id):
            writer.writerow(
                (
                    int(particle_id),
                    int(events.event_ordinal[row]),
                    events.time_s[row],
                    *events.position_m[row],
                    *events.velocity_pre_m_s[row],
                    *events.velocity_post_m_s[row],
                    events.charge_number_pre[row],
                    events.charge_number_post[row],
                    int(events.boundary_id[row]),
                    events.law_id[row],
                    events.outcome[row],
                )
            )
    return trajectory_rows, int(events.particle_id.size)


def _verify_revisions(manifest: dict[str, Any], expected: dict[str, Any]) -> None:
    keys = (
        "case_schema_version",
        "result_schema_version",
        "engine_algorithm_revision",
        "boundary_algorithm_revision",
        "result_algorithm_revision",
    )
    differences = {
        key: (manifest.get(key), expected.get(key))
        for key in keys
        if manifest.get(key) != expected.get(key)
    }
    if differences:
        raise ValueError(f"candidate revisions differ: {differences}")


def run(config_path: Path, output: Path) -> dict[str, object]:
    """Create and execute one locked P18-H candidate."""

    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    config_path = config_path.resolve()
    config_hash_before = _sha256(config_path)
    config = _load_config(config_path)
    references = _locked_references(config)
    output.mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve(), output / PRODUCER_FILENAME)
    shutil.copyfile(config_path, output / CONFIG_FILENAME)
    data_info = write(output / DATA_FILENAME, _candidate_bundle(config, config_hash_before))
    case_path = output / CASE_FILENAME
    case_path.write_text(
        yaml.safe_dump(_case_document(config, data_info.content_hash), sort_keys=False),
        encoding="utf-8",
    )
    result_path = output / RESULT_DIRECTORY
    simulate(load_case(case_path), result_path)
    result = open_result(result_path)
    manifest = dict(result.manifest)
    _verify_revisions(manifest, _mapping(config["expected_candidate"], "expected_candidate"))
    trajectory_rows, event_rows = _write_projection(output, result)
    failures = result.read_failure_events()
    final = result.read_final()
    series = result.read_lifecycle_series()
    config_hash_after = _sha256(config_path)
    if config_hash_after != config_hash_before:
        raise ValueError("configuration changed during candidate run")
    report: dict[str, object] = {
        "status": "COMPLETE",
        "tool_revision": TOOL_REVISION,
        "configuration_sha256_before": config_hash_before,
        "configuration_sha256_after": config_hash_after,
        "producer_source_sha256": _sha256(output / PRODUCER_FILENAME),
        "case_sha256": _sha256(case_path),
        "canonical_input_sha256": _sha256(output / DATA_FILENAME),
        "canonical_content_hash": data_info.content_hash,
        "result_manifest_sha256": _sha256(result_path / "run.json"),
        "trajectory_sha256": _sha256(output / TRAJECTORY_FILENAME),
        "trajectory_rows": trajectory_rows,
        "events_sha256": _sha256(output / EVENT_FILENAME),
        "event_rows": event_rows,
        "failure_event_count": int(failures.particle_id.size),
        "final_lifecycle": int(final.lifecycle[0]),
        "final_kinematics_valid": int(final.kinematics_valid[0]),
        "final_state": {
            "time_s": float(final.time_s[0]),
            "position_m": final.position_m[0].tolist(),
            "velocity_m_per_s": final.velocity_m_s[0].tolist(),
            "charge_number_e": float(final.charge_number[0]),
        },
        "lifecycle_series_final": {
            "pending": int(series.pending[-1]),
            "active": int(series.active[-1]),
            "stuck": int(series.stuck[-1]),
            "held": int(series.held[-1]),
            "escaped": int(series.escaped[-1]),
            "failed": int(series.failed[-1]),
        },
        "lifecycle_counts": manifest.get("lifecycle_counts"),
        "revisions": {
            key: manifest[key]
            for key in (
                "case_schema_version",
                "result_schema_version",
                "engine_algorithm_revision",
                "boundary_algorithm_revision",
                "result_algorithm_revision",
            )
        },
        "locked_references": references,
    }
    with (output / REPORT_FILENAME).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.config, arguments.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
