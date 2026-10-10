"""Assemble completed M3-C2 participant outputs for the ensemble evaluator.

This is a narrow external-V&V adapter.  It verifies the two runner-owned
participant manifests and emits only the campaign schema consumed by
``evaluate_m3c2_stochastic_ensemble.py``.  It does not run either solver or
change participant artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, cast

import numpy as np

from chamber_particles.case_format import read_with_info
from tools.vv.comsol.actual_run_receipt import observed_receipt_sha256
from tools.vv.comsol.meaning_preflight import require_supported_comparison
from tools.vv.comsol.normalize_m3c2_comsol_pilot import _output_times

TOOL_REVISION: Final = "m3c2_campaign_assembler_v2"
CASE_ID: Final = "M3-C2A_caseA_100nm_common-P1"
CAMPAIGN_IDENTITY_KEYS: Final = {
    "case_id",
    "evaluation_case_id",
    "output_slug",
    "final_registration_kind",
    "candidate_case_name_prefix",
}
CAMPAIGN_BINDING_KEYS: Final = {
    "contract_sha256",
    "input_sha256",
    "input_content_hash",
}
PILOT_AUTHORIZATION_KEYS: Final = {"path", "sha256"}
LEGACY_CAMPAIGN_IDENTITY: Final = {
    "case_id": "formal_iondrag_theory_consistent/caseA_100nm",
    "evaluation_case_id": CASE_ID,
    "output_slug": "caseA_100nm",
    "final_registration_kind": "m3c2_caseA_100nm_final_campaign",
    "candidate_case_name_prefix": "m3c2_caseA_100nm",
}
LEGACY_COMSOL_TOOL_REVISION: Final = "m3c2_caseA_100nm_comsol_normalizer_v2"
LEGACY_CANDIDATE_TOOL_REVISION: Final = "m3c2_caseA_100nm_candidate_campaign_runner_v3"
COMSOL_TOOL_REVISION: Final = "m3c2_comsol_campaign_normalizer_v5"
HISTORICAL_COMSOL_TOOL_REVISION: Final = "m3c2_comsol_campaign_normalizer_v4"
CANDIDATE_TOOL_REVISION: Final = "m3c2_candidate_campaign_runner_v4"
SUPPORTED_CANDIDATE_TOOL_REVISIONS: Final = {
    CANDIDATE_TOOL_REVISION,
    "m3c2_candidate_campaign_runner_v5",
    "m3c2_candidate_campaign_runner_v6",
}
PARTICLE_IDS: Final = tuple(range(1, 288))
PARTICLE_COUNT: Final = len(PARTICLE_IDS)
OUTPUT_TIMES_S: Final = tuple(_output_times())
OUTPUT_COUNT: Final = len(OUTPUT_TIMES_S)
END_TIME_S: Final = OUTPUT_TIMES_S[-1]
REPLICA_COUNTS: Final = {"pilot": 4, "final": 32}
PATH_SENSITIVITY_PURPOSE: Final = "brownian_first_passage_path_depth_sensitivity"
TRAJECTORY_COLUMNS: Final = {
    "particle_id",
    "time_s",
    "r_m",
    "z_m",
    "velocity_r_m_per_s",
    "velocity_z_m_per_s",
    "charge_number_e",
    "lifecycle",
}
EVENT_COLUMNS: Final = {"particle_id", "event_time_s", "outcome", "boundary_semantic"}
type Purpose = Literal["pilot", "final"]


@dataclass(frozen=True, slots=True)
class Artifact:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class Replica:
    seed: int
    trajectory: Artifact
    events: Artifact
    performance: Artifact | None


@dataclass(frozen=True, slots=True)
class Level:
    level_id: str
    ordinal: int
    numerical_setting: dict[str, Any]
    replicas: tuple[Replica, ...]


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def _sequence(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return cast(list[Any], value)


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _json(path: Path, name: str) -> dict[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), name)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name}: {path}") from error


def _campaign_identity_record(value: object, name: str) -> dict[str, str]:
    record = _mapping(value, name)
    if set(record) != CAMPAIGN_IDENTITY_KEYS:
        raise ValueError(f"{name} must contain exactly the registered identity fields")
    if any(
        not isinstance(record[key], str) or not record[key] or record[key] != record[key].strip()
        for key in CAMPAIGN_IDENTITY_KEYS
    ):
        raise ValueError(f"{name} values must be nonempty strings")
    identity = {key: cast(str, record[key]) for key in CAMPAIGN_IDENTITY_KEYS}
    return identity


def _participant_campaign_identity(
    comsol: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, str]:
    comsol_value = comsol.get("campaign_identity")
    candidate_value = candidate.get("campaign_identity")
    if comsol_value is None and candidate_value is None:
        if (
            comsol.get("tool_revision") != LEGACY_COMSOL_TOOL_REVISION
            or comsol.get("case_id") != LEGACY_CAMPAIGN_IDENTITY["case_id"]
            or candidate.get("tool_revision") != LEGACY_CANDIDATE_TOOL_REVISION
        ):
            raise ValueError("new participant manifests must define campaign identity")
        return dict(LEGACY_CAMPAIGN_IDENTITY)
    if comsol_value is None or candidate_value is None:
        raise ValueError("campaign identity must be present on both participant manifests")
    if comsol.get("tool_revision") not in {COMSOL_TOOL_REVISION, HISTORICAL_COMSOL_TOOL_REVISION}:
        raise ValueError("COMSOL participant tool revision is not supported")
    if candidate.get("tool_revision") not in SUPPORTED_CANDIDATE_TOOL_REVISIONS:
        raise ValueError("candidate participant tool revision is not supported")
    comsol_identity = _campaign_identity_record(comsol_value, "COMSOL campaign identity")
    candidate_identity = _campaign_identity_record(candidate_value, "candidate campaign identity")
    if comsol_identity != candidate_identity:
        raise ValueError("participant campaign identities differ")
    if comsol.get("case_id") != comsol_identity["evaluation_case_id"]:
        raise ValueError("COMSOL case_id differs from the campaign identity")
    return candidate_identity


def _campaign_binding_record(value: object, name: str) -> dict[str, str]:
    record = _mapping(value, name)
    if set(record) != CAMPAIGN_BINDING_KEYS:
        raise ValueError(f"{name} must contain exactly the registered binding fields")
    if any(not isinstance(record[key], str) for key in CAMPAIGN_BINDING_KEYS):
        raise ValueError(f"{name} values must be strings")
    result = {key: cast(str, record[key]).lower() for key in CAMPAIGN_BINDING_KEYS}
    for key in ("contract_sha256", "input_sha256"):
        digest = result[key]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"{name}.{key} is not a SHA-256 digest")
    content_digest = result["input_content_hash"].removeprefix("sha256:")
    if (
        not result["input_content_hash"].startswith("sha256:")
        or len(content_digest) != 64
        or any(character not in "0123456789abcdef" for character in content_digest)
    ):
        raise ValueError(f"{name}.input_content_hash is invalid")
    return result


def _participant_campaign_binding(
    comsol: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, str] | None:
    comsol_value = comsol.get("campaign_binding")
    candidate_value = candidate.get("campaign_binding")
    if comsol_value is None and candidate_value is None:
        if comsol.get("campaign_identity") is None and candidate.get("campaign_identity") is None:
            return None
        raise ValueError("new participant manifests must define campaign binding")
    if comsol_value is None or candidate_value is None:
        raise ValueError("campaign binding must be present on both participant manifests")
    comsol_binding = _campaign_binding_record(comsol_value, "COMSOL campaign binding")
    candidate_binding = _campaign_binding_record(candidate_value, "candidate campaign binding")
    if comsol_binding != candidate_binding:
        raise ValueError("participant campaign bindings differ")
    return candidate_binding


def _candidate_evaluation_policy_sha256(
    candidate: dict[str, Any], campaign_identity: dict[str, str]
) -> str | None:
    value = candidate.get("evaluation_policy_sha256")
    if value is None:
        if campaign_identity != LEGACY_CAMPAIGN_IDENTITY:
            raise ValueError("new candidate manifests must define evaluation policy SHA-256")
        return None
    if not isinstance(value, str) or value != value.lower():
        raise ValueError("candidate evaluation policy SHA-256 must be lowercase hexadecimal")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("candidate evaluation policy SHA-256 is invalid")
    return value


def _pilot_authorization_record(value: object, name: str) -> dict[str, str]:
    record = _mapping(value, name)
    if set(record) != PILOT_AUTHORIZATION_KEYS:
        raise ValueError(f"{name} must contain exactly path and sha256")
    if not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str):
        raise ValueError(f"{name} values must be strings")
    path = Path(cast(str, record["path"]))
    if path.is_absolute() or str(path) in {"", "."} or ".." in path.parts:
        raise ValueError(f"{name}.path must be a repository-relative path")
    digest = cast(str, record["sha256"]).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name}.sha256 is not a SHA-256 digest")
    return {"path": path.as_posix(), "sha256": digest}


def _participant_pilot_authorization(
    comsol: dict[str, Any],
    candidate: dict[str, Any],
    campaign_identity: dict[str, str],
    purpose: Purpose,
) -> dict[str, str] | None:
    if purpose != "pilot":
        return None
    comsol_value = comsol.get("pilot_authorization")
    candidate_value = candidate.get("pilot_authorization")
    required = campaign_identity != LEGACY_CAMPAIGN_IDENTITY
    if comsol_value is None and candidate_value is None:
        if required:
            raise ValueError("new pilot participant manifests must define pilot authorization")
        return None
    if comsol_value is None or candidate_value is None:
        raise ValueError("pilot authorization must be present on both participant manifests")
    comsol_authorization = _pilot_authorization_record(comsol_value, "COMSOL pilot authorization")
    candidate_authorization = _pilot_authorization_record(
        candidate_value, "candidate pilot authorization"
    )
    if comsol_authorization != candidate_authorization:
        raise ValueError("participant pilot authorizations differ")
    return candidate_authorization


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path_value: object, hash_value: object, root: Path, name: str) -> Artifact:
    relative = Path(str(path_value))
    if relative.is_absolute() or str(relative) in {"", "."}:
        raise ValueError(f"{name}.path must be a nonempty relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{name} is missing or outside its participant directory: {path}")
    expected = str(hash_value).lower()
    if len(expected) != 64 or _sha256(path) != expected:
        raise ValueError(f"{name} SHA-256 differs")
    return Artifact(path, expected)


def _nested_artifact(record: object, root: Path, name: str) -> Artifact:
    value = _mapping(record, name)
    return _artifact(value.get("path"), value.get("sha256"), root, name)


def _flat_artifact(record: dict[str, Any], key: str, root: Path, name: str) -> Artifact:
    return _artifact(record.get(key), record.get(f"{key}_sha256"), root, name)


def _time_index(value: str, times_s: tuple[float, ...], path: Path, line: int) -> int:
    try:
        time_s = float(value)
    except ValueError as error:
        raise ValueError(f"{path}:{line}: invalid time_s") from error
    insertion = int(np.searchsorted(times_s, time_s))
    candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(times_s)]
    if not candidates:
        raise ValueError(f"{path}:{line}: time_s is outside the campaign schedule")
    closest = min(candidates, key=lambda index: abs(times_s[index] - time_s))
    tolerance = max(2.0e-14, 64.0 * abs(float(np.spacing(times_s[closest]))))
    if not math.isfinite(time_s) or abs(times_s[closest] - time_s) > tolerance:
        raise ValueError(f"{path}:{line}: unexpected output time")
    return closest


def _validate_trajectory_scope(
    path: Path, particle_ids: tuple[int, ...], times_s: tuple[float, ...]
) -> int:
    particles = {particle_id: index for index, particle_id in enumerate(particle_ids)}
    seen = np.zeros((len(times_s), len(particle_ids)), dtype=np.bool_)
    rows = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if TRAJECTORY_COLUMNS.difference(reader.fieldnames or ()):
            raise ValueError(f"{path}: trajectory columns are incomplete")
        for line, row in enumerate(reader, start=2):
            try:
                particle = particles[int(row["particle_id"])]
            except (KeyError, ValueError) as error:
                raise ValueError(f"{path}:{line}: particle is outside campaign scope") from error
            time = _time_index(row["time_s"], times_s, path, line)
            if seen[time, particle]:
                raise ValueError(f"{path}:{line}: duplicate particle/time row")
            seen[time, particle] = True
            rows += 1
    if not bool(np.all(seen)):
        raise ValueError(f"{path}: trajectory does not cover the complete particle/time scope")
    return rows


def _validate_event_scope(path: Path, particle_ids: tuple[int, ...], end_time_s: float) -> int:
    allowed = set(particle_ids)
    seen: set[int] = set()
    rows = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if EVENT_COLUMNS.difference(reader.fieldnames or ()):
            raise ValueError(f"{path}: event columns are incomplete")
        for line, row in enumerate(reader, start=2):
            try:
                particle_id = int(row["particle_id"])
                event_time_s = float(row["event_time_s"])
            except (KeyError, ValueError) as error:
                raise ValueError(f"{path}:{line}: invalid terminal event") from error
            if particle_id not in allowed or particle_id in seen:
                raise ValueError(f"{path}:{line}: duplicate or out-of-scope terminal event")
            if not math.isfinite(event_time_s) or not 0.0 <= event_time_s <= end_time_s:
                raise ValueError(f"{path}:{line}: terminal-event time is outside the run")
            seen.add(particle_id)
            rows += 1
    return rows


def _validate_row_count(value: object, observed: int, name: str) -> None:
    if _integer(value, name) != observed:
        raise ValueError(f"{name} differs from the artifact row count")


def _comsol_replica(raw: object, root: Path, level_id: str, index: int) -> Replica:
    name = f"comsol.{level_id}.replicas[{index}]"
    record = _mapping(raw, name)
    if record.get("status") not in {None, "COMPLETE"}:
        raise ValueError(f"{name} is not complete")
    trajectory = _nested_artifact(record.get("trajectory"), root, f"{name}.trajectory")
    events = _nested_artifact(record.get("events"), root, f"{name}.events")
    trajectory_rows = _validate_trajectory_scope(trajectory.path, PARTICLE_IDS, OUTPUT_TIMES_S)
    event_rows = _validate_event_scope(events.path, PARTICLE_IDS, END_TIME_S)
    _validate_row_count(record.get("trajectory_rows"), trajectory_rows, f"{name}.trajectory_rows")
    _validate_row_count(record.get("event_count"), event_rows, f"{name}.event_count")
    performance = (
        None
        if record.get("performance") is None
        else _nested_artifact(record["performance"], root, f"{name}.performance")
    )
    return Replica(_integer(record.get("seed"), f"{name}.seed"), trajectory, events, performance)


def _candidate_replica(raw: object, root: Path, level_id: str, index: int) -> Replica:
    name = f"candidate.{level_id}.replicas[{index}]"
    record = _mapping(raw, name)
    if record.get("status") != "COMPLETE" or record.get("participant") != "candidate":
        raise ValueError(f"{name} is not a completed candidate cell")
    trajectory = _flat_artifact(record, "trajectory", root, f"{name}.trajectory")
    events = _flat_artifact(record, "events", root, f"{name}.events")
    trajectory_rows = _validate_trajectory_scope(trajectory.path, PARTICLE_IDS, OUTPUT_TIMES_S)
    event_rows = _validate_event_scope(events.path, PARTICLE_IDS, END_TIME_S)
    _validate_row_count(record.get("trajectory_rows"), trajectory_rows, f"{name}.trajectory_rows")
    _validate_row_count(record.get("event_rows"), event_rows, f"{name}.event_rows")
    performance = (
        None
        if record.get("performance") is None
        else _flat_artifact(record, "performance", root, f"{name}.performance")
    )
    return Replica(_integer(record.get("seed"), f"{name}.seed"), trajectory, events, performance)


def _replicas(values: object, root: Path, level_id: str, participant: str) -> tuple[Replica, ...]:
    loader = _comsol_replica if participant == "comsol" else _candidate_replica
    replicas = tuple(
        sorted(
            (
                loader(value, root, level_id, index)
                for index, value in enumerate(
                    _sequence(values, f"{participant}.{level_id}.replicas")
                )
            ),
            key=lambda replica: replica.seed,
        )
    )
    seeds = [replica.seed for replica in replicas]
    if any(seed < 0 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError(f"{participant}.{level_id} seeds must be nonnegative and unique")
    return replicas


def _comsol_levels(raw: dict[str, Any], root: Path, purpose: Purpose) -> tuple[Level, ...]:
    levels: list[Level] = []
    for index, value in enumerate(_sequence(raw.get("levels"), "comsol.levels")):
        record = _mapping(value, f"comsol.levels[{index}]")
        setting = dict(_mapping(record.get("numerical_setting"), "comsol numerical_setting"))
        if purpose == "pilot":
            if setting.get("purpose") not in {None, "macro_step_convergence"}:
                raise ValueError("COMSOL pilot level has an unexpected purpose")
            setting["purpose"] = "macro_step_convergence"
        level_id = str(record.get("level_id", ""))
        levels.append(
            Level(
                level_id,
                _integer(record.get("ordinal"), f"comsol.{level_id}.ordinal"),
                setting,
                _replicas(record.get("replicas"), root, level_id, "comsol"),
            )
        )
    levels.sort(key=lambda level: level.ordinal)
    return tuple(levels)


def _ordered_candidate_records(
    raw: dict[str, Any], purpose: Purpose, path_reference_level_id: str | None
) -> list[tuple[str, dict[str, Any]]]:
    raw_records = _mapping(raw.get("levels"), "candidate.levels")
    records = [
        (level_id, _mapping(value, f"candidate.levels.{level_id}"))
        for level_id, value in raw_records.items()
    ]
    if purpose == "pilot":
        macro = [item for item in records if item[1].get("purpose") == "macro_step_convergence"]
        path = [item for item in records if item[1].get("purpose") == PATH_SENSITIVITY_PURPOSE]
        macro.sort(key=lambda item: -float(item[1].get("dt_s", math.nan)))
        if len(macro) != 3 or len(path) != 1 or len(records) != 4:
            raise ValueError("candidate pilot must contain three macro levels and one path level")
        if path_reference_level_id not in {name for name, _ in macro}:
            raise ValueError("candidate pilot path reference must explicitly name one macro level")
        return [*macro, path[0]]
    if path_reference_level_id is not None:
        raise ValueError("candidate path reference is only valid for a pilot campaign")
    if len(records) != 1:
        raise ValueError("candidate final campaign must contain exactly one level")
    return records


def _candidate_levels(
    raw: dict[str, Any],
    root: Path,
    purpose: Purpose,
    path_reference_level_id: str | None,
) -> tuple[Level, ...]:
    ordered = _ordered_candidate_records(raw, purpose, path_reference_level_id)
    levels: list[Level] = []
    for ordinal, (level_id, record) in enumerate(ordered):
        setting = {key: item for key, item in record.items() if key != "replicas"}
        if setting.get("purpose") == PATH_SENSITIVITY_PURPOSE:
            setting["reference_level_id"] = path_reference_level_id
        levels.append(
            Level(
                level_id,
                ordinal,
                setting,
                _replicas(record.get("replicas"), root, level_id, "candidate"),
            )
        )
    return tuple(levels)


def _validate_level_identity(
    participant: str, levels: tuple[Level, ...], expected_levels: int
) -> None:
    if len(levels) != expected_levels:
        raise ValueError(f"{participant} has the wrong number of numerical levels")
    if [level.ordinal for level in levels] != list(range(len(levels))):
        raise ValueError(f"{participant} level ordinals must run from zero coarse-to-fine")
    level_ids = [level.level_id for level in levels]
    if any(not level_id for level_id in level_ids) or len(level_ids) != len(set(level_ids)):
        raise ValueError(f"{participant} level IDs must be nonempty and unique")


def _participant_seed_set(
    participant: str, levels: tuple[Level, ...], expected_replicas: int
) -> set[int]:
    seeds_by_level = [{replica.seed for replica in level.replicas} for level in levels]
    if any(len(seeds) != expected_replicas for seeds in seeds_by_level):
        raise ValueError(f"{participant} has the wrong replica count")
    if any(seeds != seeds_by_level[0] for seeds in seeds_by_level[1:]):
        raise ValueError(f"{participant} levels do not reuse one seed set")
    return seeds_by_level[0]


def _participant_artifacts(levels: tuple[Level, ...]) -> set[Path]:
    artifacts = [
        artifact
        for level in levels
        for replica in level.replicas
        for artifact in (replica.trajectory.path, replica.events.path)
    ]
    if len(artifacts) != len(set(artifacts)):
        raise ValueError("trajectory and event artifacts must be unique per campaign cell")
    return set(artifacts)


def _validate_levels(participants: dict[str, tuple[Level, ...]], purpose: Purpose) -> None:
    expected_levels = (
        {"comsol": 3, "candidate": 4}
        if purpose == "pilot"
        else {
            "comsol": 1,
            "candidate": 1,
        }
    )
    expected_replicas = REPLICA_COUNTS[purpose]
    seed_sets: dict[str, set[int]] = {}
    artifact_paths: set[Path] = set()
    for participant, levels in participants.items():
        _validate_level_identity(participant, levels, expected_levels[participant])
        seeds = _participant_seed_set(participant, levels, expected_replicas)
        artifacts = _participant_artifacts(levels)
        if artifact_paths & artifacts:
            raise ValueError("trajectory and event artifacts must be unique per campaign cell")
        artifact_paths.update(artifacts)
        seed_sets[participant] = seeds
    if seed_sets["comsol"] & seed_sets["candidate"]:
        raise ValueError("participant seed sets must be disjoint")


def _canonical_scope(canonical_input: Path, candidate: dict[str, Any]) -> tuple[list[float], str]:
    if not canonical_input.is_file():
        raise ValueError(f"canonical input is missing: {canonical_input}")
    if candidate.get("input_sha256") != _sha256(canonical_input):
        raise ValueError("candidate manifest canonical-input SHA-256 differs")
    data, info = read_with_info(canonical_input)
    if candidate.get("input_content_hash") != info.content_hash:
        raise ValueError("candidate manifest canonical-input content hash differs")
    sources = [source for source in data.sources if source.name == "particles"]
    if len(sources) != 1 or tuple(int(value) for value in sources[0].particle_id) != PARTICLE_IDS:
        raise ValueError("canonical input particle IDs are not exactly 1..287")
    nodes = np.asarray(data.geometry.nodes_m, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 2 or not np.isfinite(nodes).all():
        raise ValueError("canonical input geometry nodes are invalid")
    lower = np.min(nodes, axis=0)
    upper = np.max(nodes, axis=0)
    bounds = [float(lower[0]), float(upper[0]), float(lower[1]), float(upper[1])]
    if not bounds[0] < bounds[1] or not bounds[2] < bounds[3]:
        raise ValueError("canonical input geometry bounds are degenerate")
    return bounds, info.content_hash


def _validate_manifest_scope(comsol: dict[str, Any], candidate: dict[str, Any]) -> None:
    if (
        tuple(float(value) for value in comsol.get("common_observation_times", ()))
        != OUTPUT_TIMES_S
    ):
        raise ValueError("COMSOL participant does not use the canonical 121-point schedule")
    if (
        int(candidate.get("particle_count", -1)) != PARTICLE_COUNT
        or int(candidate.get("output_count", -1)) != OUTPUT_COUNT
        or float(candidate.get("time_end_s", math.nan)) != END_TIME_S
    ):
        raise ValueError("candidate participant time/particle scope differs")


def _performance_is_comparable(artifact: Artifact | None) -> bool:
    if artifact is None:
        return False
    record = _json(artifact.path, "participant performance")
    status = record.get("measurement_status")
    if status not in {None, "MEASURED", "MEASURED_PROCESS_LIFETIME_HIGH_WATER"}:
        return False
    if record.get("non_authoritative_reason") is not None:
        return False
    try:
        return (
            math.isfinite(float(record["wall_time_s"]))
            and float(record["wall_time_s"]) > 0.0
            and int(record["peak_rss_bytes"]) > 0
            and int(record["output_bytes"]) >= 0
            and int(record["particle_count"]) == PARTICLE_COUNT
            and int(record["output_frames"]) == OUTPUT_COUNT
        )
    except (KeyError, TypeError, ValueError):
        return False


def _relative_artifact(artifact: Artifact, output_root: Path) -> dict[str, str]:
    return {
        "path": Path(os.path.relpath(artifact.path, output_root)).as_posix(),
        "sha256": artifact.sha256,
    }


def _render_participants(
    participants: dict[str, tuple[Level, ...]], output_root: Path
) -> dict[str, object]:
    replicas = [
        replica
        for levels in participants.values()
        for level in levels
        for replica in level.replicas
    ]
    performance_paths = [
        replica.performance.path for replica in replicas if replica.performance is not None
    ]
    include_performance = (
        len(performance_paths) == len(replicas)
        and len(performance_paths) == len(set(performance_paths))
        and all(_performance_is_comparable(replica.performance) for replica in replicas)
    )
    rendered: dict[str, object] = {}
    for participant, levels in participants.items():
        rendered[participant] = {
            "levels": [
                {
                    "level_id": level.level_id,
                    "ordinal": level.ordinal,
                    "numerical_setting": level.numerical_setting,
                    "replicas": [
                        {
                            "seed": replica.seed,
                            "trajectory": _relative_artifact(replica.trajectory, output_root),
                            "events": _relative_artifact(replica.events, output_root),
                            **(
                                {
                                    "performance": _relative_artifact(
                                        cast(Artifact, replica.performance), output_root
                                    )
                                }
                                if include_performance
                                else {}
                            ),
                        }
                        for replica in level.replicas
                    ],
                }
                for level in levels
            ]
        }
    return rendered


def require_participant_meaning(
    comsol: dict[str, Any], candidate: dict[str, Any], comsol_root: Path
) -> None:
    """Bind new certification to this campaign's model, input, and actual runs."""
    if (
        comsol.get("tool_revision") != COMSOL_TOOL_REVISION
        and candidate.get("tool_revision") != "m3c2_candidate_campaign_runner_v6"
    ):
        return
    binding = _participant_campaign_binding(comsol, candidate)
    model_digest = comsol.get("source_model_sha256")
    if not isinstance(model_digest, str) or len(model_digest) != 64:
        raise ValueError("meaning_preflight_inventory requires this run's source model SHA-256")
    if binding is None:
        raise ValueError("meaning_preflight_inventory requires canonical campaign binding")
    observed = frozenset(
        observed_receipt_sha256(replica.get("actual_run_readback"), comsol_root)
        for level in _sequence(comsol.get("levels"), "COMSOL levels")
        for replica in _sequence(_mapping(level, "COMSOL level").get("replicas"), "replicas")
    )
    require_supported_comparison(
        comsol.get("meaning_preflight_inventory"),
        comsol_root,
        expected_model_sha256=model_digest,
        expected_field_identity=binding["input_content_hash"],
        required_observed_sha256=observed,
    )


def _participant_manifest_projection(
    comsol: dict[str, Any],
    candidate: dict[str, Any],
    comsol_root: Path,
    candidate_root: Path,
    purpose: Purpose,
    output_root: Path,
    candidate_path_reference_level_id: str | None,
) -> dict[str, object]:
    if (comsol.get("schema_version"), comsol.get("manifest_kind"), comsol.get("participant")) != (
        1,
        "m3c2_participant",
        "comsol",
    ) or comsol.get("status") != "COMPLETE_NORMALIZED_NOT_EVALUATED":
        raise ValueError("COMSOL participant manifest is not complete normalized M3-C2 output")
    expected_comparison = f"READY_FOR_INDEPENDENT_ENSEMBLE_{purpose.upper()}_EVALUATION"
    if (
        candidate.get("participant") != "candidate"
        or candidate.get("status") != "COMPLETE"
        or candidate.get("comparison_status") != expected_comparison
    ):
        raise ValueError("candidate participant manifest is not complete")
    campaign_identity = _participant_campaign_identity(comsol, candidate)
    campaign_binding = _participant_campaign_binding(comsol, candidate)
    require_participant_meaning(comsol, candidate, comsol_root)
    evaluation_policy_sha256 = _candidate_evaluation_policy_sha256(candidate, campaign_identity)
    pilot_authorization = _participant_pilot_authorization(
        comsol, candidate, campaign_identity, purpose
    )
    _validate_manifest_scope(comsol, candidate)
    participants = {
        "comsol": _comsol_levels(comsol, comsol_root, purpose),
        "candidate": _candidate_levels(
            candidate,
            candidate_root,
            purpose,
            candidate_path_reference_level_id,
        ),
    }
    _validate_levels(participants, purpose)
    return {
        "case_id": campaign_identity["evaluation_case_id"],
        "campaign_binding": campaign_binding,
        "evaluation_policy_sha256": evaluation_policy_sha256,
        "pilot_authorization": pilot_authorization,
        "participants": _render_participants(participants, output_root),
    }


def assemble_campaign(
    comsol_manifest: Path,
    candidate_manifest: Path,
    canonical_input: Path,
    purpose: Purpose,
    output: Path,
    *,
    candidate_path_reference_level_id: str | None = None,
) -> dict[str, object]:
    """Validate participant outputs and write one evaluator-ready campaign manifest."""

    if purpose not in REPLICA_COUNTS:
        raise ValueError("campaign purpose must be pilot or final")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"campaign manifest already exists: {output}")
    comsol_path = comsol_manifest.resolve()
    candidate_path = candidate_manifest.resolve()
    comsol = _json(comsol_path, "COMSOL participant manifest")
    candidate = _json(candidate_path, "candidate participant manifest")
    projection = _participant_manifest_projection(
        comsol,
        candidate,
        comsol_path.parent,
        candidate_path.parent,
        purpose,
        output.parent,
        candidate_path_reference_level_id,
    )
    campaign_binding = cast(dict[str, str] | None, projection["campaign_binding"])
    resolved_input = canonical_input.resolve()
    bounds, content_hash = _canonical_scope(resolved_input, candidate)
    if campaign_binding is not None and (
        campaign_binding["input_sha256"] != _sha256(resolved_input)
        or campaign_binding["input_content_hash"] != content_hash
    ):
        raise ValueError("campaign binding differs from the canonical input")
    campaign: dict[str, object] = {
        "schema_version": 1,
        "manifest_kind": "m3c2_campaign",
        "case_id": projection["case_id"],
        "purpose": purpose,
        **({"campaign_binding": campaign_binding} if campaign_binding is not None else {}),
        **(
            {"evaluation_policy_sha256": projection["evaluation_policy_sha256"]}
            if projection["evaluation_policy_sha256"] is not None
            else {}
        ),
        **(
            {"pilot_authorization": projection["pilot_authorization"]}
            if projection["pilot_authorization"] is not None
            else {}
        ),
        "scope": {
            "particle_ids": list(PARTICLE_IDS),
            "output_times_s": list(OUTPUT_TIMES_S),
            "geometry_bounds_m": bounds,
            "canonical_input": {
                "path": Path(os.path.relpath(resolved_input, output.parent)).as_posix(),
                "sha256": _sha256(resolved_input),
                "content_hash": content_hash,
            },
        },
        "participant_manifests": {
            "comsol": {
                "path": Path(os.path.relpath(comsol_path, output.parent)).as_posix(),
                "sha256": _sha256(comsol_path),
            },
            "candidate": {
                "path": Path(os.path.relpath(candidate_path, output.parent)).as_posix(),
                "sha256": _sha256(candidate_path),
            },
        },
        "participants": projection["participants"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return campaign


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comsol-manifest", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--canonical-input", required=True, type=Path)
    parser.add_argument("--purpose", required=True, choices=tuple(REPLICA_COUNTS))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--candidate-path-reference-level-id")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    campaign = assemble_campaign(
        arguments.comsol_manifest,
        arguments.candidate_manifest,
        arguments.canonical_input,
        cast(Purpose, arguments.purpose),
        arguments.output,
        candidate_path_reference_level_id=arguments.candidate_path_reference_level_id,
    )
    print(json.dumps(campaign, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
