"""Run the locked M3-C1 common-P1 first-material-event candidate.

This external V&V utility extends only the already accepted fine-step
candidate case.  It uses the public solver API and does not add a benchmark
mode to the production engine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml

from chamber_particles import load_case, open_result, simulate

TOOL_REVISION: Final = "m3c1_common_p1_material_event_candidate_v2"
CASE_FILENAME: Final = "candidate_material_event.yaml"
RESULT_DIRECTORY: Final = "result_material_event"
TRAJECTORY_FILENAME: Final = "candidate_trajectory.csv"
PRE_EVENT_TRAJECTORY_FILENAME: Final = "candidate_pre_event_trajectory.csv"
EVENT_FILENAME: Final = "candidate_events.csv"
RUN_REPORT_FILENAME: Final = "candidate_run_report.json"
PREPARE_REPORT_FILENAME: Final = "prepare_report.json"
PRODUCER_SOURCE_FILENAME: Final = "run_m3c1_material_event_candidate.py"
_LIFECYCLE: Final = ("pending", "active", "stuck", "escaped", "failed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _solver_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_json_exclusive(path: Path, payload: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _stage_producer_source(output: Path) -> dict[str, str]:
    source = Path(__file__).resolve()
    staged = output / PRODUCER_SOURCE_FILENAME
    with source.open("rb") as source_stream, staged.open("xb") as staged_stream:
        for block in iter(lambda: source_stream.read(1024 * 1024), b""):
            staged_stream.write(block)
    digest = _sha256(staged)
    if digest != _sha256(source):
        raise OSError("staged candidate producer differs from the executing source")
    return {"file": PRODUCER_SOURCE_FILENAME, "sha256": digest}


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _load_config(path: Path) -> dict[str, Any]:
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), "configuration")
    required = {
        "schema_version",
        "evaluation_id",
        "evaluation_revision",
        "classification",
        "expected_comsol_version",
        "source_model",
        "case",
        "candidate",
        "reference",
        "expected_first_event",
        "acceptance",
        "claim_policy",
    }
    if set(payload) != required:
        raise ValueError(f"configuration keys differ: {sorted(set(payload) ^ required)}")
    if payload["schema_version"] != 1 or payload["evaluation_revision"] != 1:
        raise ValueError("unsupported material-event configuration revision")
    case = _mapping(payload["case"], "case")
    if case.get("workflow") != "caseA" or case.get("particle_count") != 287:
        raise ValueError("material-event anchor must select Case-A and 287 particles")
    if not math.isclose(float(case.get("diameter_m", 0.0)), 1.0e-7):
        raise ValueError("material-event anchor must select 100 nm particles")
    if not math.isclose(float(case.get("pre_event_end_s", 0.0)), 4.5e-4):
        raise ValueError("material-event predecessor must end at 450 us")
    if not math.isclose(float(case.get("event_window_end_s", 0.0)), 4.5875e-4):
        raise ValueError("material-event window must end at 458.75 us")
    if not math.isclose(float(case.get("fixed_rk4_step_s", 0.0)), 1.5625e-7):
        raise ValueError("material-event anchor must use the accepted fine step")
    output = np.asarray(case.get("output_times_s"), dtype=np.float64)
    if output.shape != (48,) or output[0] != 0.0 or output[-1] != 4.5875e-4:
        raise ValueError("material-event output schedule must contain the locked 48 frames")
    if bool((np.diff(output) <= 0.0).any()):
        raise ValueError("material-event output schedule must be strictly increasing")
    expected = _mapping(payload["expected_first_event"], "expected_first_event")
    if expected != {
        "particle_id": 57,
        "candidate_boundary_id": 6,
        "candidate_external_id": 134,
        "boundary_group": "wafer",
        "comsol_status_code": 3,
        "candidate_lifecycle": "stuck",
        "candidate_law": "stick",
        "wafer_z_m": 0.022,
    }:
        raise ValueError("first-event identity differs from the preregistered anchor")
    return payload


def _locked_path(root: Path, relative: object, expected_hash: object, name: str) -> Path:
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise ValueError(f"{name} lock is malformed")
    path = (root / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"locked {name} is missing: {path}")
    if _sha256(path) != expected_hash:
        raise ValueError(f"locked {name} hash differs: {path}")
    return path


def _locked_inputs(config: dict[str, Any]) -> dict[str, Path]:
    root = _solver_root()
    candidate = _mapping(config["candidate"], "candidate")
    parent_root = (root / str(candidate["parent_root_relative_path"])).resolve()
    if not parent_root.is_dir():
        raise FileNotFoundError(f"candidate parent root is missing: {parent_root}")
    paths = {
        "input": _locked_path(
            parent_root,
            candidate["input_filename"],
            candidate["input_sha256"],
            "candidate input",
        ),
        "case": _locked_path(
            parent_root,
            candidate["parent_case_filename"],
            candidate["parent_case_sha256"],
            "candidate parent case",
        ),
        "trajectory": _locked_path(
            parent_root,
            candidate["parent_trajectory_filename"],
            candidate["parent_trajectory_sha256"],
            "candidate parent trajectory",
        ),
        "manifest": _locked_path(
            parent_root,
            candidate["parent_result_manifest_relative_path"],
            candidate["parent_result_manifest_sha256"],
            "candidate parent result manifest",
        ),
        "run_report": _locked_path(
            parent_root,
            candidate["parent_run_report_filename"],
            candidate["parent_run_report_sha256"],
            "candidate parent run report",
        ),
    }
    reference = _mapping(config["reference"], "reference")
    pre_event_root = (root / str(reference["pre_event_root_relative_path"])).resolve()
    paths["reference_receipt"] = _locked_path(
        pre_event_root,
        "common_p1_run_receipt.json",
        reference["pre_event_run_receipt_sha256"],
        "common-P1 pre-event receipt",
    )
    paths["comparison"] = _locked_path(
        root,
        reference["pre_event_comparison_relative_path"],
        reference["pre_event_comparison_sha256"],
        "common-P1 pre-event comparison",
    )
    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8-sig"))
    if comparison.get("status") != "PASS":
        raise ValueError("locked common-P1 pre-event comparison is not PASS")
    return paths


def prepare(config_path: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    config_hash_before = _sha256(config_path)
    config = _load_config(config_path)
    locked = _locked_inputs(config)
    parent = _mapping(yaml.safe_load(locked["case"].read_text(encoding="utf-8")), "case file")
    case = _mapping(parent.get("case"), "case file.case")
    time = _mapping(parent.get("time"), "case file.time")
    trajectories = _mapping(
        _mapping(parent.get("output"), "case file.output").get("trajectories"),
        "case file.output.trajectories",
    )
    if (
        time != {"start_s": 0.0, "end_s": 4.5e-4, "dt_s": 1.5625e-7}
        or case.get("expected_content_hash") != config["candidate"]["input_content_hash"]
    ):
        raise ValueError("locked parent case is not the accepted fine-step common-P1 case")
    output.mkdir(parents=True)
    producer_source = _stage_producer_source(output)
    case["name"] = "m3c1_caseA_100nm_common_p1_first_material_event"
    case["data_path"] = Path(os.path.relpath(locked["input"], output)).as_posix()
    time["end_s"] = float(config["case"]["event_window_end_s"])
    trajectories["schedule"] = {"explicit_times_s": config["case"]["output_times_s"]}
    case_path = output / CASE_FILENAME
    with case_path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(parent, stream, sort_keys=False)
    loaded = load_case(case_path)
    if loaded.content_hash != config["candidate"]["input_content_hash"]:
        raise ValueError("prepared case does not resolve the locked canonical input")
    config_hash_after = _sha256(config_path)
    if config_hash_after != config_hash_before:
        raise ValueError("material-event configuration changed during candidate preparation")
    report: dict[str, object] = {
        "status": "PREPARED",
        "tool_revision": TOOL_REVISION,
        "configuration": str(config_path.resolve()),
        "configuration_sha256": config_hash_before,
        "configuration_sha256_before": config_hash_before,
        "configuration_sha256_after": config_hash_after,
        "producer_source": producer_source,
        "case": CASE_FILENAME,
        "case_sha256": _sha256(case_path),
        "locked_inputs": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in locked.items()
        },
        "scope": {
            "particles": int(config["case"]["particle_count"]),
            "frames": len(config["case"]["output_times_s"]),
            "time_window_s": [0.0, float(config["case"]["event_window_end_s"])],
            "dt_s": float(config["case"]["fixed_rk4_step_s"]),
        },
        "claim_policy": config["claim_policy"],
    }
    _write_json_exclusive(output / PREPARE_REPORT_FILENAME, report)
    return report


def _write_trajectory(path: Path, result: object, *, end_s: float | None = None) -> int:
    rows = 0
    with path.open("x", encoding="utf-8", newline="") as stream:
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
            if end_s is not None and frame.time_s > end_s:
                continue
            for index, particle_id in enumerate(frame.particle_id):
                writer.writerow(
                    (
                        int(particle_id),
                        frame.time_s,
                        frame.position_m[index, 0],
                        frame.position_m[index, 1],
                        frame.velocity_m_s[index, 0],
                        frame.velocity_m_s[index, 1],
                        frame.charge_number[index],
                        _LIFECYCLE[int(frame.lifecycle[index])],
                    )
                )
                rows += 1
    return rows


def _write_events(path: Path, result: object) -> int:
    events = result.read_boundary_events()  # type: ignore[attr-defined]
    with path.open("x", encoding="utf-8", newline="") as stream:
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
                "charge_number_pre_e",
                "charge_number_post_e",
                "primary_facet_id",
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
                    events.position_m[row, 0],
                    events.position_m[row, 1],
                    events.normal[row, 0],
                    events.normal[row, 1],
                    events.velocity_pre_m_s[row, 0],
                    events.velocity_pre_m_s[row, 1],
                    events.velocity_post_m_s[row, 0],
                    events.velocity_post_m_s[row, 1],
                    events.charge_number_pre[row],
                    events.charge_number_post[row],
                    int(events.primary_facet_id[row]),
                    int(events.boundary_id[row]),
                    events.law_id[row],
                    events.outcome[row],
                )
            )
    return int(events.particle_id.size)


def _prepared_run_identity(config_path: Path, output: Path) -> tuple[str, Path, str, Path, str]:
    config_hash_before = _sha256(config_path)
    prepare_path = output / PREPARE_REPORT_FILENAME
    prepared = json.loads(prepare_path.read_text(encoding="utf-8"))
    if prepared.get("tool_revision") != TOOL_REVISION:
        raise ValueError("prepared candidate tool revision differs")
    if prepared.get("configuration_sha256") != config_hash_before or any(
        prepared.get(name) != config_hash_before
        for name in ("configuration_sha256_before", "configuration_sha256_after")
    ):
        raise ValueError("prepared candidate configuration hash differs")
    case_path = output / CASE_FILENAME
    case_hash_before = _sha256(case_path)
    if prepared.get("case_sha256") != case_hash_before:
        raise ValueError("prepared candidate case hash differs")
    producer = _mapping(prepared.get("producer_source"), "prepared producer_source")
    staged_source = output / PRODUCER_SOURCE_FILENAME
    source_hash_before = _sha256(staged_source)
    if producer != {"file": PRODUCER_SOURCE_FILENAME, "sha256": source_hash_before}:
        raise ValueError("prepared candidate producer source differs")
    if source_hash_before != _sha256(Path(__file__).resolve()):
        raise ValueError("executing candidate producer differs from its staged source")
    return config_hash_before, case_path, case_hash_before, staged_source, source_hash_before


def _new_run_outputs(output: Path) -> tuple[Path, Path, Path, Path, Path]:
    outputs = (
        output / RESULT_DIRECTORY,
        output / TRAJECTORY_FILENAME,
        output / PRE_EVENT_TRAJECTORY_FILENAME,
        output / EVENT_FILENAME,
        output / RUN_REPORT_FILENAME,
    )
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"candidate output already exists: {existing[0]}")
    return outputs


def _first_event(events: Any, event_rows: int) -> dict[str, object] | None:
    if not event_rows:
        return None
    row = int(np.argmin(events.time_s))
    return {
        "particle_id": int(events.particle_id[row]),
        "event_time_s": float(events.time_s[row]),
        "hit_position_m": events.position_m[row].tolist(),
        "velocity_pre_m_s": events.velocity_pre_m_s[row].tolist(),
        "velocity_post_m_s": events.velocity_post_m_s[row].tolist(),
        "charge_number_pre_e": float(events.charge_number_pre[row]),
        "charge_number_post_e": float(events.charge_number_post[row]),
        "primary_facet_id": int(events.primary_facet_id[row]),
        "boundary_id": int(events.boundary_id[row]),
        "law": str(events.law_id[row]),
        "outcome": str(events.outcome[row]),
    }


def _verify_run_inputs_unchanged(
    config_path: Path,
    config_hash_before: str,
    case_path: Path,
    case_hash_before: str,
    staged_source: Path,
    source_hash_before: str,
) -> tuple[str, str, str]:
    config_hash_after = _sha256(config_path)
    case_hash_after = _sha256(case_path)
    source_hash_after = _sha256(staged_source)
    if config_hash_after != config_hash_before:
        raise ValueError("material-event configuration changed during candidate execution")
    if case_hash_after != case_hash_before:
        raise ValueError("prepared candidate case changed during candidate execution")
    if source_hash_after != source_hash_before:
        raise ValueError("staged candidate producer changed during candidate execution")
    return config_hash_after, case_hash_after, source_hash_after


def run(config_path: Path, output: Path) -> dict[str, object]:
    config = _load_config(config_path)
    locked = _locked_inputs(config)
    config_hash_before, case_path, case_hash_before, staged_source, source_hash_before = (
        _prepared_run_identity(config_path, output)
    )
    prepare_path = output / PREPARE_REPORT_FILENAME
    result_path, trajectory_path, pre_event_trajectory_path, event_path, report_path = (
        _new_run_outputs(output)
    )
    simulate(load_case(case_path), result_path)
    result = open_result(result_path)
    trajectory_rows = _write_trajectory(trajectory_path, result)
    pre_event_rows = _write_trajectory(
        pre_event_trajectory_path,
        result,
        end_s=float(config["case"]["pre_event_end_s"]),
    )
    event_rows = _write_events(event_path, result)
    failures = result.read_failure_events()
    events = result.read_boundary_events()
    first_event = _first_event(events, event_rows)
    config_hash_after, case_hash_after, source_hash_after = _verify_run_inputs_unchanged(
        config_path,
        config_hash_before,
        case_path,
        case_hash_before,
        staged_source,
        source_hash_before,
    )
    prepare_hash_after = _sha256(prepare_path)
    report: dict[str, object] = {
        "status": "COMPLETE" if result.manifest.get("status") == "complete" else "INCOMPLETE",
        "tool_revision": TOOL_REVISION,
        "configuration_sha256": config_hash_before,
        "configuration_sha256_before": config_hash_before,
        "configuration_sha256_after": config_hash_after,
        "producer_source": {"file": PRODUCER_SOURCE_FILENAME, "sha256": source_hash_after},
        "prepare_report_sha256": prepare_hash_after,
        "case_sha256": case_hash_after,
        "result_manifest_sha256": _sha256(result_path / "run.json"),
        "result_manifest_status": result.manifest.get("status"),
        "trajectory": TRAJECTORY_FILENAME,
        "trajectory_sha256": _sha256(trajectory_path),
        "trajectory_rows": trajectory_rows,
        "pre_event_trajectory": PRE_EVENT_TRAJECTORY_FILENAME,
        "pre_event_trajectory_sha256": _sha256(pre_event_trajectory_path),
        "pre_event_trajectory_rows": pre_event_rows,
        "events": EVENT_FILENAME,
        "events_sha256": _sha256(event_path),
        "event_rows": event_rows,
        "first_event": first_event,
        "failure_event_count": int(failures.particle_id.size),
        "lifecycle_counts": result.manifest.get("lifecycle_counts"),
        "boundary_interactions": result.manifest.get("boundary_interactions"),
        "event_refinement": result.manifest.get("event_refinement"),
        "pre_event_lineage": {
            "historical_parent_trajectory_sha256": _sha256(locked["trajectory"]),
            "bitwise_parent_equality_required": False,
            "acceptance_owner": "evaluate_m3c1_common_field.py",
        },
        "revisions": {
            name: value
            for name, value in result.manifest.items()
            if name.endswith("_revision") and value is not None
        },
        "claim_policy": config["claim_policy"],
    }
    _write_json_exclusive(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("config", type=Path)
        subparser.add_argument("output", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    config = arguments.config.resolve()
    output = arguments.output.resolve()
    report = prepare(config, output) if arguments.command == "prepare" else run(config, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") in {"PREPARED", "COMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
