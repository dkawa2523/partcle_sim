"""Read observed COMSOL run snapshots and resolve terminal boundary evidence.

Readback values are evidence for the external meaning preflight, not a physical
equivalence verdict. Missing readback or event-cause observations stay untested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from chamber_particles.case_format import read_with_info
from tools.vv.comsol.boundary_response_mapping import (
    ActualBoundaryResponse,
    Outcome,
    TerminalBoundaryResolution,
    resolve_terminal_boundary,
)
from tools.vv.comsol.meaning_preflight import load_inventory

RECEIPT_NAME = "actual_binding_receipt.json"
RECEIPT_REVISION = "comsol_particle_actual_readback_v1"
READBACK_PREFIX = "M3C_ACTUAL|directory="
_OUTCOMES = {"Stick": "stuck", "Freeze": "held", "Disappear": "escaped", "Bounce": "active"}


@dataclass(frozen=True, slots=True)
class ActualRunReceipt:
    """Raw observed facts and the separately observed terminal causes."""

    artifact: dict[str, Any]
    responses: dict[int, ActualBoundaryResponse]
    terminals: dict[int, dict[str, Any]]
    responses_complete: bool = False

    def terminal(self, particle_id: int, outcome: str, time_s: float) -> TerminalBoundaryResolution:
        observation = self.terminals.get(particle_id)
        ids: list[int] = []
        excluded = False
        if observation is not None:
            if observation["outcome"] != outcome or not math.isclose(
                observation["event_time_s"], time_s, rel_tol=0.0, abs_tol=3.0e-13
            ):
                raise ValueError("Actual terminal observation differs from the status history")
            ids = observation["boundary_ids"]
            excluded = observation["other_terminal_causes_excluded"]
        responses = self.responses if ids or self.responses_complete else {}
        return resolve_terminal_boundary(cast(Outcome, outcome), ids, responses, excluded)


def _object(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location}: expected an object")
    return cast(dict[str, Any], value)


def _ids(value: object, location: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int or item < 1 for item in value):
        raise ValueError(f"{location}: expected positive boundary IDs")
    if len(value) != len(set(value)):
        raise ValueError(f"{location}: duplicate boundary ID")
    return value


def _observed_property(feature: dict[str, Any], name: str) -> object:
    properties = _object(feature.get("properties"), "feature.properties")
    entry = _object(properties.get(name), f"feature.properties.{name}")
    value = entry.get("observed_value") if entry.get("observation") == "OBSERVED" else None
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    return value


def _responses(companion: dict[str, Any]) -> tuple[dict[int, ActualBoundaryResponse], bool]:
    features = _object(companion.get("boundary_features"), "companion.boundary_features")
    result: dict[int, ActualBoundaryResponse] = {}
    conflicting: set[int] = set()
    seen: set[int] = set()
    complete = True
    for raw in features.values():
        feature = _object(raw, "boundary feature")
        if type(feature.get("active")) is not bool:
            return {}, False
        if feature["active"] is not True:
            continue
        if feature.get("selection_observation") != "OBSERVED":
            return {}, False
        ids = _ids(feature.get("boundary_ids"), "boundary feature.boundary_ids")
        conflicting.update(seen.intersection(ids))
        seen.update(ids)
        law = _observed_property(feature, "WallCondition")
        group = feature.get("semantic_group")
        if (
            not isinstance(law, str)
            or law not in _OUTCOMES
            or not isinstance(group, str)
            or not group
        ):
            complete = False
            continue
        response = ActualBoundaryResponse(cast(Outcome, _OUTCOMES[law]), group)
        for boundary_id in ids:
            result[boundary_id] = response
    # Feature selection is not evidence of override priority. Even identical
    # competing laws leave the effective feature unknown.
    return {
        key: value for key, value in result.items() if key not in conflicting
    }, complete and not conflicting


def _terminals(raw: object) -> dict[int, dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("terminal_observations: expected an array")
    result: dict[int, dict[str, Any]] = {}
    for entry in raw:
        observation = _object(entry, "terminal observation")
        particle_id = observation.get("particle_id")
        time_s = observation.get("event_time_s")
        if type(particle_id) is not int or particle_id < 1 or particle_id in result:
            raise ValueError("terminal observation: invalid or duplicate particle ID")
        if isinstance(time_s, bool) or not isinstance(time_s, (float, int)):
            raise ValueError("terminal observation: invalid event time")
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("terminal observation: invalid event time")
        if observation.get("outcome") not in ("held", "stuck", "escaped"):
            raise ValueError("terminal observation: invalid outcome")
        _ids(observation.get("boundary_ids"), "terminal observation.boundary_ids")
        excluded = observation.get("other_terminal_causes_excluded")
        if type(excluded) is not bool:
            raise ValueError("terminal observation: cause exclusion must be boolean")
        if excluded and not observation.get("cause_evidence"):
            raise ValueError("terminal observation: cause exclusion evidence is missing")
        result[particle_id] = observation
    return result


def write_boundary_meaning(candidate: Path, output: Path) -> Path:
    """Materialize canonical boundary-ID meanings for the external adapter."""

    data, info = read_with_info(candidate)
    groups: dict[str, str] = {}
    for boundary_id, group_id in zip(
        data.geometry.boundary.boundary_id, data.geometry.boundary.group_id, strict=True
    ):
        key = str(int(boundary_id))
        group = data.geometry.group_names[int(group_id)]
        if key in groups and groups[key] != group:
            raise ValueError("A native boundary ID has conflicting canonical meanings")
        groups[key] = group
    path = output / "boundary_meaning.json"
    path.write_text(
        json.dumps(
            {
                "canonical_input_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "canonical_content_hash": info.content_hash,
                "boundary_groups": groups,
                "authority": "canonical_geometry_boundary_ids_and_groups",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _native_terminal_values(row: list[str], path: Path) -> tuple[int, int, float, int]:
    if len(row) != 4:
        raise ValueError(f"{path}: expected four native terminal columns")
    values = [float(value) for value in row]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}: nonfinite terminal evidence")
    particle_id, code, event_time, boundary_id = values
    if any(value != int(value) for value in (particle_id, code, boundary_id)):
        raise ValueError(f"{path}: noninteger native identity/status")
    if particle_id < 1 or int(code) not in (1, 2, 3, 4) or boundary_id < 0:
        raise ValueError(f"{path}: invalid native particle, status, or boundary")
    return int(particle_id), int(code), event_time, int(boundary_id)


def _native_terminal_probe(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    seen: set[int] = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(line for line in stream if not line.startswith("%"))
        for row in rows:
            particle_id, code, event_time, boundary_id = _native_terminal_values(row, path)
            if particle_id in seen:
                raise ValueError(f"{path}: duplicate native particle")
            seen.add(particle_id)
            if code == 1:
                continue
            observed = code in (2, 3) and boundary_id > 0
            result[particle_id] = {
                "particle_id": particle_id,
                "outcome": {2: "held", 3: "stuck", 4: "escaped"}[code],
                "event_time_s": event_time,
                "boundary_ids": [boundary_id] if observed else [],
                "other_terminal_causes_excluded": observed,
                "cause_evidence": "native bndenv(dom) at the retained terminal boundary"
                if observed
                else "NOT_TESTED: disappeared position has no boundary environment",
            }
    return _terminals(list(result.values()))


def read_actual_run_receipt(
    directory: Path, boundary_meaning: Path | None = None
) -> ActualRunReceipt:
    """Read one receipt, retaining unavailable historical observations as such."""

    path = directory / RECEIPT_NAME
    if not path.is_file():
        return ActualRunReceipt(
            {"observation": "NOT_TESTED", "reason": "Actual run readback was not exported."},
            {},
            {},
        )
    raw = _object(load_inventory(path), str(path))
    if (
        type(raw.get("schema_version")) is not int
        or raw.get("schema_version") != 1
        or raw.get("tool_revision") != RECEIPT_REVISION
    ):
        raise ValueError(f"{path}: unsupported actual readback schema or revision")
    for snapshot in ("source", "companion"):
        _object(raw.get(snapshot), f"{path}.{snapshot}")
    artifact = {
        "path": RECEIPT_NAME,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "observation": "EXPORTED",
        "assembly_rhs": "NOT_TESTED",
        "boundary_event_export": raw.get("boundary_event_export", "NOT_TESTED"),
    }
    responses, complete = _responses(raw["companion"])
    if boundary_meaning is not None and boundary_meaning.is_file():
        meaning = _object(load_inventory(boundary_meaning), str(boundary_meaning))
        groups = _object(meaning.get("boundary_groups"), "canonical boundary groups")
        response_count = len(responses)
        responses = {
            boundary_id: ActualBoundaryResponse(response.outcome, groups[str(boundary_id)])
            for boundary_id, response in responses.items()
            if str(boundary_id) in groups and isinstance(groups[str(boundary_id)], str)
        }
        complete = complete and len(responses) == response_count
        artifact["canonical_boundary_meaning"] = {
            "sha256": hashlib.sha256(boundary_meaning.read_bytes()).hexdigest(),
            "canonical_content_hash": meaning.get("canonical_content_hash"),
        }
    terminals = _terminals(raw.get("terminal_observations", []))
    probe = directory / "terminal_boundary_raw.csv"
    if probe.is_file():
        if terminals:
            raise ValueError("Two independent terminal observation owners are present")
        terminals = _native_terminal_probe(probe)
        artifact["terminal_boundary_probe"] = {
            "path": probe.name,
            "sha256": hashlib.sha256(probe.read_bytes()).hexdigest(),
        }
        artifact["boundary_event_export"] = "NATIVE_BOUNDARY_ID_FOR_RETAINED_TERMINALS_ONLY"
    return ActualRunReceipt(artifact, responses, terminals, complete)


def terminal_evidence(particle_id: int, resolution: TerminalBoundaryResolution) -> dict[str, Any]:
    return {
        "particle_id": particle_id,
        "classification": resolution.classification,
        "identification": resolution.identification,
        "observed_boundary_ids": list(resolution.observed_boundary_ids),
        "semantic_group": resolution.semantic_group,
        "reason": resolution.reason,
    }


def normalize_terminal_event(
    actual: ActualRunReceipt, particle_id: int, outcome: str, time_s: float
) -> tuple[tuple[object, ...], dict[str, Any]]:
    """Keep status observations while naming a boundary only when supported."""

    resolution = actual.terminal(particle_id, outcome, time_s)
    event = (
        particle_id,
        time_s,
        "terminal_boundary" if resolution.classification == "SUPPORTED" else "terminal_status",
        outcome,
        resolution.semantic_group or "",
    )
    return event, terminal_evidence(particle_id, resolution)


def inventory_artifact(directory: Path) -> dict[str, str] | None:
    """Attach the preregistered inventory without certifying its own declarations."""

    path = directory / "meaning_inventory.json"
    if not path.is_file():
        return None
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def observed_receipt_sha256(reference: object, base_directory: Path) -> str:
    """Require the normalized run's actual readback bytes, not a declared status."""
    if not isinstance(reference, dict) or reference.get("observation") != "EXPORTED":
        raise ValueError("actual_run_readback must identify an exported actual receipt")
    relative_path = reference.get("path")
    digest = reference.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(digest, str):
        raise ValueError("actual_run_readback requires path and sha256")
    path = (base_directory / relative_path).resolve()
    if (
        path.name != RECEIPT_NAME
        or not path.is_file()
        or hashlib.sha256(path.read_bytes()).hexdigest() != digest
    ):
        raise ValueError("actual_run_readback artifact differs from its SHA-256")
    actual = read_actual_run_receipt(path.parent)
    if actual.artifact["sha256"] != digest:
        raise ValueError("actual_run_readback artifact identity differs")
    return digest


def materialize_actual_run_receipts(root: Path) -> None:
    """Persist raw API JSON emitted through COMSOL's permitted process log."""

    seen: set[Path] = set()
    for line in (
        (root / "comsol_process.log").read_text(encoding="utf-8-sig", errors="strict").splitlines()
    ):
        if not line.startswith(READBACK_PREFIX):
            continue
        directory, separator, payload = line[len(READBACK_PREFIX) :].partition("|json=")
        if not separator:
            raise ValueError("Malformed actual readback log record")
        target = (root / directory / RECEIPT_NAME).resolve()
        if not target.is_relative_to(root.resolve()) or target in seen:
            raise ValueError("Actual readback path escapes the run or is duplicated")
        seen.add(target)
        encoded = (payload + "\n").encode("utf-8")
        if target.exists():
            if target.read_bytes() != encoded:
                raise ValueError("Actual readback log differs from the existing immutable receipt")
        else:
            with target.open("xb") as stream:
                stream.write(encoded)
        load_inventory(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    materialize_actual_run_receipts(parser.parse_args().run_directory.resolve())


if __name__ == "__main__":
    main()
