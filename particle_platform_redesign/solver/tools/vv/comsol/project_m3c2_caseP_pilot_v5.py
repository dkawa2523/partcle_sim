"""Create the single immutable v5 projection of the completed Case-P pilot.

No solver output is generated or rewritten.  The tool recursively verifies the
v4 assembled campaign and its participant artifacts, checks that v5 preserves
the registered science policy, and changes only the evaluation-policy hash.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from tools.vv.comsol import evaluate_m3c2_stochastic_ensemble as ensemble

TOOL_REVISION: Final = "m3c2_caseP_pilot_v5_projection_v1"
PROJECTION_KIND: Final = "m3c2_caseP_policy_only_campaign_projection"
PROJECTION_REASON: Final = "final前のscope/hash検証補強"
POLICY_ID: Final = "M3-C2A-caseP-100nm-ensemble-evaluation"


def _mapping(value: object, location: str) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    raise ValueError(f"{location} must be a mapping")


def _sequence(value: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, location: str) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), location)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _policy_science_payload(raw: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(raw)
    for key in ("policy_revision", "classification", "migration", "independence_evidence"):
        payload.pop(key, None)
    return payload


def _validate_policy_transition(v4: dict[str, Any], v5: dict[str, Any]) -> None:
    if (
        v4.get("policy_id"),
        v4.get("policy_revision"),
        v5.get("policy_id"),
        v5.get("policy_revision"),
    ) != (POLICY_ID, 4, POLICY_ID, 5):
        raise ValueError("projection requires Case-P policy revisions 4 and 5")
    migration = _mapping(v5.get("migration"), "v5 migration")
    if (
        migration.get("reason") != PROJECTION_REASON
        or migration.get("raw_solver_outputs_reused_without_rerun") is not True
        or migration.get("thresholds_seeds_observables_unchanged") is not True
    ):
        raise ValueError("v5 migration declaration differs")
    if _policy_science_payload(v4) != _policy_science_payload(v5):
        raise ValueError("v5 changes registered thresholds, seeds, observables, or claims")


def _resolve_nested_artifact(base: Path, path_value: str) -> Path:
    relative = Path(path_value)
    candidates = (
        [relative] if relative.is_absolute() else [base / relative, _repository_root() / relative]
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise ValueError(f"participant artifact is missing: {path_value}")


def _verify_artifact(base: Path, path_value: str, sha256: str, location: str) -> None:
    path = _resolve_nested_artifact(base, path_value)
    if _sha256(path) != sha256.lower():
        raise ValueError(f"{location} SHA-256 differs")


def _verify_nested_artifacts(value: object, base: Path, location: str = "participant") -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _verify_nested_artifacts(item, base, f"{location}[{index}]")
        return
    if not isinstance(value, dict):
        return
    if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
        _verify_artifact(base, value["path"], value["sha256"], location)
    for key, digest in value.items():
        if not key.endswith("_sha256") or not isinstance(digest, str):
            continue
        owner = key.removesuffix("_sha256")
        path_value = value.get(owner)
        if isinstance(path_value, str):
            _verify_artifact(base, path_value, digest, f"{location}.{owner}")
    for key, item in value.items():
        _verify_nested_artifacts(item, base, f"{location}.{key}")


def _participant_paths(source: Path, raw: dict[str, Any]) -> list[Path]:
    records = _mapping(raw.get("participant_manifests"), "participant manifests")
    if set(records) != {"comsol", "candidate"}:
        raise ValueError("source campaign participant manifest set differs")
    paths: list[Path] = []
    for participant in ("comsol", "candidate"):
        record = _mapping(records[participant], f"{participant} participant manifest")
        path = _resolve_nested_artifact(source.parent, str(record.get("path")))
        if _sha256(path) != str(record.get("sha256", "")).lower():
            raise ValueError(f"{participant} participant manifest SHA-256 differs")
        paths.append(path)
    return paths


def _relative_artifact(path: Path, base: Path) -> dict[str, str]:
    relative = os.path.relpath(path, base).replace("\\", "/")
    return {"path": relative, "sha256": _sha256(path)}


def _projected_document(
    source: Path,
    source_raw: dict[str, Any],
    v4_policy: Path,
    v5_policy: Path,
    output: Path,
) -> dict[str, Any]:
    if output.parent.resolve() != source.parent.parent.resolve():
        raise ValueError("projection must be a sibling of the immutable source campaign directory")
    projected = copy.deepcopy(source_raw)
    source_policy_sha256 = _sha256(v4_policy)
    target_policy_sha256 = _sha256(v5_policy)
    if projected.get("evaluation_policy_sha256") != source_policy_sha256:
        raise ValueError("source campaign was not assembled for the locked v4 policy")
    projected["evaluation_policy_sha256"] = target_policy_sha256
    unchanged = copy.deepcopy(source_raw)
    unchanged.pop("evaluation_policy_sha256", None)
    projected["policy_projection"] = {
        "schema_version": 1,
        "projection_kind": PROJECTION_KIND,
        "tool_revision": TOOL_REVISION,
        "reason": PROJECTION_REASON,
        "source_campaign": _relative_artifact(source, output),
        "source_policy": _relative_artifact(v4_policy, output),
        "target_policy": _relative_artifact(v5_policy, output),
        "changed_fields": ["evaluation_policy_sha256"],
        "unchanged_payload_sha256": _object_sha256(unchanged),
        "raw_solver_outputs_reused_without_rerun": True,
    }
    return projected


def project(source: Path, v4_policy: Path, v5_policy: Path, output: Path) -> dict[str, object]:
    source = source.resolve()
    v4_policy = v4_policy.resolve()
    v5_policy = v5_policy.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"v5 projection output already exists: {output}")
    source_raw = _load_json(source, "v4 source campaign")
    v4_raw = _load_json(v4_policy, "v4 evaluation policy")
    v5_raw = _load_json(v5_policy, "v5 evaluation policy")
    _validate_policy_transition(v4_raw, v5_raw)
    v4 = ensemble._load_policy(v4_policy)
    if v4.revision != 4:
        raise ValueError("source projection policy is not revision 4")
    ensemble._load_campaign(source, v4)
    for manifest_path in _participant_paths(source, source_raw):
        _verify_nested_artifacts(
            _load_json(manifest_path, f"participant manifest {manifest_path.name}"),
            manifest_path.parent,
        )
    projected = _projected_document(source, source_raw, v4_policy, v5_policy, output)
    output.mkdir(parents=True)
    target = output / "campaign.json"
    target.write_text(json.dumps(projected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "status": "PROJECTED",
        "tool_revision": TOOL_REVISION,
        "source_campaign_sha256": _sha256(source),
        "campaign": str(target),
        "campaign_sha256": _sha256(target),
        "raw_solver_outputs_reused_without_rerun": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_campaign", type=Path)
    parser.add_argument("v4_policy", type=Path)
    parser.add_argument("v5_policy", type=Path)
    parser.add_argument("output", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = project(
        arguments.source_campaign,
        arguments.v4_policy,
        arguments.v5_policy,
        arguments.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
