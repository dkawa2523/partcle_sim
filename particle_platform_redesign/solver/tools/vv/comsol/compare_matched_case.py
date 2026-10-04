"""Compare one preregistered deterministic COMSOL/solver matched case.

This module is an external V&V tool.  It consumes normalized CSV artifacts and
never changes, configures, or imports the production trajectory engine.  A
successful comparison means only that the supplied artifacts are within the
manifest's preregistered tolerances.  It is not a claim that COMSOL is a golden
truth or that the two solvers have equal numerical accuracy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import yaml

TOOL_REVISION: Final = "m3v_matched_case_comparison_v1"
_LAYER_ORDER: Final = ("field", "force", "integrator", "trajectory", "boundary")
_READINESS_STATUSES: Final = frozenset({"CONFIRMED", "BLOCKED", "NOT_APPLICABLE"})
_ACCURACY_STATUSES: Final = frozenset({"CONFIRMED", "NOT_TESTED", "NOT_APPLICABLE"})
_SHA256_LENGTH: Final = 64

Record = dict[str, object]


class ManifestError(ValueError):
    """The matched-case manifest is incomplete or internally inconsistent."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ManifestError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class _Table:
    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ManifestError(f"{label} must be a string-keyed mapping")
    return value


def _sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ManifestError(f"{label} must be a list")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{label} must be a nonempty string")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ManifestError(f"{label} must be a boolean")
    return value


def _nonnegative_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ManifestError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ManifestError(f"{label} must be finite and nonnegative")
    return result


def _exact_keys(mapping: dict[str, object], expected: set[str], label: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise ManifestError(
            f"{label} keys must be exactly {sorted(expected)}; got {sorted(actual)}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_matched_case_manifest(path: str | Path) -> dict[str, object]:
    """Load and strictly validate one matched-case comparison manifest."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.load(manifest_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ManifestError(f"cannot read matched-case manifest: {error}") from error
    manifest = _mapping(raw, "manifest")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "comparison_id",
            "classification",
            "claim_policy",
            "readiness",
            "setting_alignment",
            "artifacts",
            "layers",
            "accuracy_evidence",
        },
        "manifest",
    )
    if manifest["schema_version"] != 1:
        raise ManifestError("schema_version must be 1")
    _text(manifest["comparison_id"], "comparison_id")
    if manifest["classification"] != "external_comsol_matched_case_diagnostic":
        raise ManifestError("classification must be external_comsol_matched_case_diagnostic")
    if manifest["claim_policy"] != "within_preregistered_tolerances_only":
        raise ManifestError("claim_policy must be within_preregistered_tolerances_only")
    _validate_readiness(manifest["readiness"])
    _validate_settings(manifest["setting_alignment"])
    artifact_names = _validate_artifacts(manifest["artifacts"])
    _validate_layers(manifest["layers"], artifact_names)
    _validate_accuracy_evidence(manifest["accuracy_evidence"])
    return manifest


def _validate_readiness(value: object) -> None:
    rows = _sequence(value, "readiness")
    names: set[str] = set()
    for index, value_row in enumerate(rows):
        row = _mapping(value_row, f"readiness[{index}]")
        _exact_keys(
            row, {"name", "required", "status", "evidence", "reason"}, f"readiness[{index}]"
        )
        name = _text(row["name"], f"readiness[{index}].name")
        if name in names:
            raise ManifestError(f"duplicate readiness name: {name}")
        names.add(name)
        _boolean(row["required"], f"readiness[{index}].required")
        status = _text(row["status"], f"readiness[{index}].status")
        if status not in _READINESS_STATUSES:
            raise ManifestError(f"unsupported readiness status: {status}")
        _text(row["evidence"], f"readiness[{index}].evidence")
        _text(row["reason"], f"readiness[{index}].reason")


def _validate_settings(value: object) -> None:
    rows = _sequence(value, "setting_alignment")
    names: set[str] = set()
    for index, value_row in enumerate(rows):
        row = _mapping(value_row, f"setting_alignment[{index}]")
        _exact_keys(
            row,
            {
                "name",
                "required_value",
                "reference_value",
                "candidate_value",
                "reference_evidence",
                "candidate_evidence",
            },
            f"setting_alignment[{index}]",
        )
        name = _text(row["name"], f"setting_alignment[{index}].name")
        if name in names:
            raise ManifestError(f"duplicate setting name: {name}")
        names.add(name)
        _text(row["reference_evidence"], f"setting_alignment[{index}].reference_evidence")
        _text(row["candidate_evidence"], f"setting_alignment[{index}].candidate_evidence")
        for field in ("required_value", "reference_value", "candidate_value"):
            try:
                json.dumps(row[field], allow_nan=False, sort_keys=True)
            except (TypeError, ValueError) as error:
                raise ManifestError(f"setting {name}.{field} is not finite JSON data") from error


def _validate_artifacts(value: object) -> set[str]:
    artifacts = _mapping(value, "artifacts")
    _exact_keys(artifacts, {"reference", "candidate"}, "artifacts")
    sides = {
        side: _mapping(artifacts[side], f"artifacts.{side}") for side in ("reference", "candidate")
    }
    if set(sides["reference"]) != set(sides["candidate"]):
        raise ManifestError("reference and candidate artifact names must match")
    for side, records in sides.items():
        for name, value_record in records.items():
            _text(name, f"artifacts.{side} artifact name")
            record = _mapping(value_record, f"artifacts.{side}.{name}")
            _exact_keys(record, {"path", "sha256"}, f"artifacts.{side}.{name}")
            path_value = record["path"]
            hash_value = record["sha256"]
            if (path_value is None) != (hash_value is None):
                raise ManifestError(
                    f"artifacts.{side}.{name} path and sha256 must both be null or both be set"
                )
            if path_value is not None:
                _text(path_value, f"artifacts.{side}.{name}.path")
                digest = _text(hash_value, f"artifacts.{side}.{name}.sha256").casefold()
                if len(digest) != _SHA256_LENGTH or any(
                    character not in "0123456789abcdef" for character in digest
                ):
                    raise ManifestError(f"artifacts.{side}.{name}.sha256 is not SHA-256")
    return set(sides["reference"])


def _validate_layers(value: object, artifact_names: set[str]) -> None:
    layers = _mapping(value, "layers")
    _exact_keys(layers, set(_LAYER_ORDER), "layers")
    used_artifacts: set[str] = set()
    for layer_name in _LAYER_ORDER:
        layer = _mapping(layers[layer_name], f"layers.{layer_name}")
        _exact_keys(
            layer,
            {"required", "artifact", "keys", "numeric", "categorical"},
            f"layers.{layer_name}",
        )
        _boolean(layer["required"], f"layers.{layer_name}.required")
        artifact = _text(layer["artifact"], f"layers.{layer_name}.artifact")
        if artifact not in artifact_names:
            raise ManifestError(f"layers.{layer_name} names an unknown artifact: {artifact}")
        if artifact in used_artifacts:
            raise ManifestError(f"artifact {artifact} is assigned to more than one layer")
        used_artifacts.add(artifact)
        _validate_keys(layer["keys"], layer_name)
        _validate_numeric_specs(layer["numeric"], layer_name)
        _validate_categorical_specs(layer["categorical"], layer_name)


def _validate_keys(value: object, layer_name: str) -> None:
    rows = _sequence(value, f"layers.{layer_name}.keys")
    if not rows:
        raise ManifestError(f"layers.{layer_name}.keys must not be empty")
    names: set[str] = set()
    for index, value_row in enumerate(rows):
        row = _mapping(value_row, f"layers.{layer_name}.keys[{index}]")
        _exact_keys(
            row,
            {"name", "reference_column", "candidate_column", "kind"},
            f"layers.{layer_name}.keys[{index}]",
        )
        name = _text(row["name"], f"layers.{layer_name}.keys[{index}].name")
        if name in names:
            raise ManifestError(f"duplicate key name in {layer_name}: {name}")
        names.add(name)
        _text(row["reference_column"], f"layers.{layer_name}.keys[{index}].reference_column")
        _text(row["candidate_column"], f"layers.{layer_name}.keys[{index}].candidate_column")
        kind = _text(row["kind"], f"layers.{layer_name}.keys[{index}].kind")
        if kind not in {"text", "integer", "float"}:
            raise ManifestError(f"unsupported key kind in {layer_name}: {kind}")


def _validate_numeric_specs(value: object, layer_name: str) -> None:
    rows = _sequence(value, f"layers.{layer_name}.numeric")
    names: set[str] = set()
    for index, value_row in enumerate(rows):
        row = _mapping(value_row, f"layers.{layer_name}.numeric[{index}]")
        _exact_keys(
            row,
            {
                "name",
                "reference_column",
                "candidate_column",
                "unit",
                "absolute_tolerance",
                "relative_tolerance",
                "scale",
            },
            f"layers.{layer_name}.numeric[{index}]",
        )
        name = _text(row["name"], f"layers.{layer_name}.numeric[{index}].name")
        if name in names:
            raise ManifestError(f"duplicate numeric quantity in {layer_name}: {name}")
        names.add(name)
        for field in ("reference_column", "candidate_column", "unit"):
            _text(row[field], f"layers.{layer_name}.numeric[{index}].{field}")
        for field in ("absolute_tolerance", "relative_tolerance", "scale"):
            _nonnegative_float(row[field], f"layers.{layer_name}.numeric[{index}].{field}")


def _validate_categorical_specs(value: object, layer_name: str) -> None:
    rows = _sequence(value, f"layers.{layer_name}.categorical")
    names: set[str] = set()
    for index, value_row in enumerate(rows):
        row = _mapping(value_row, f"layers.{layer_name}.categorical[{index}]")
        _exact_keys(
            row,
            {"name", "reference_column", "candidate_column"},
            f"layers.{layer_name}.categorical[{index}]",
        )
        name = _text(row["name"], f"layers.{layer_name}.categorical[{index}].name")
        if name in names:
            raise ManifestError(f"duplicate categorical quantity in {layer_name}: {name}")
        names.add(name)
        _text(row["reference_column"], f"layers.{layer_name}.categorical[{index}].reference_column")
        _text(row["candidate_column"], f"layers.{layer_name}.categorical[{index}].candidate_column")


def _validate_accuracy_evidence(value: object) -> None:
    rows = _sequence(value, "accuracy_evidence")
    names: set[str] = set()
    for index, value_row in enumerate(rows):
        row = _mapping(value_row, f"accuracy_evidence[{index}]")
        _exact_keys(row, {"name", "status", "evidence", "reason"}, f"accuracy_evidence[{index}]")
        name = _text(row["name"], f"accuracy_evidence[{index}].name")
        if name in names:
            raise ManifestError(f"duplicate accuracy evidence name: {name}")
        names.add(name)
        status = _text(row["status"], f"accuracy_evidence[{index}].status")
        if status not in _ACCURACY_STATUSES:
            raise ManifestError(f"unsupported accuracy evidence status: {status}")
        _text(row["evidence"], f"accuracy_evidence[{index}].evidence")
        _text(row["reason"], f"accuracy_evidence[{index}].reason")


def preflight_matched_case(
    manifest_path: str | Path,
    artifact_root: str | Path,
) -> dict[str, object]:
    """Verify readiness, setting alignment, artifact presence, and hashes."""

    path = Path(manifest_path).expanduser().resolve()
    root = Path(artifact_root).expanduser().resolve()
    manifest = load_matched_case_manifest(path)
    readiness_rows, readiness_blockers = _preflight_readiness(manifest["readiness"])
    setting_rows, setting_blockers = _preflight_settings(manifest["setting_alignment"])
    verified_artifacts, artifact_blockers = _preflight_artifacts(manifest, root)
    blockers = [*readiness_blockers, *setting_blockers, *artifact_blockers]
    return {
        "tool_revision": TOOL_REVISION,
        "report_kind": "matched_case_preflight",
        "comparison_id": manifest["comparison_id"],
        "manifest_path": str(path),
        "manifest_sha256": _sha256(path),
        "artifact_root": str(root),
        "status": "PASS" if not blockers else "BLOCKED",
        "ready_for_comparison": not blockers,
        "readiness": readiness_rows,
        "setting_alignment": setting_rows,
        "verified_artifacts": verified_artifacts,
        "blockers": blockers,
        "claim": _claim("insufficient_evidence" if blockers else "comparison_not_run"),
    }


def _preflight_readiness(value: object) -> tuple[list[Record], list[Record]]:
    rows: list[Record] = []
    blockers: list[Record] = []
    for value_row in _sequence(value, "readiness"):
        row = _mapping(value_row, "readiness item")
        rows.append(dict(row))
        if bool(row["required"]) and row["status"] != "CONFIRMED":
            blockers.append(
                {
                    "category": "readiness",
                    "name": row["name"],
                    "status": row["status"],
                    "reason": row["reason"],
                }
            )
    return rows, blockers


def _preflight_settings(value: object) -> tuple[list[Record], list[Record]]:
    rows: list[Record] = []
    blockers: list[Record] = []
    for value_row in _sequence(value, "setting_alignment"):
        row = _mapping(value_row, "setting item")
        aligned = (
            row["reference_value"] == row["required_value"]
            and row["candidate_value"] == row["required_value"]
        )
        rows.append(
            {
                "name": row["name"],
                "status": "PASS" if aligned else "FAIL",
                "required_value": row["required_value"],
                "reference_value": row["reference_value"],
                "candidate_value": row["candidate_value"],
                "reference_evidence": row["reference_evidence"],
                "candidate_evidence": row["candidate_evidence"],
            }
        )
        if not aligned:
            blockers.append(
                {
                    "category": "setting_alignment",
                    "name": row["name"],
                    "status": "FAIL",
                    "reason": "reference and candidate must both equal required_value",
                }
            )
    return rows, blockers


def _preflight_artifacts(
    manifest: dict[str, object], root: Path
) -> tuple[list[Record], list[Record]]:
    verified_artifacts: list[Record] = []
    blockers: list[Record] = []
    layers = _mapping(manifest["layers"], "layers")
    artifacts = _mapping(manifest["artifacts"], "artifacts")
    for layer_name in _LAYER_ORDER:
        layer = _mapping(layers[layer_name], f"layers.{layer_name}")
        artifact_name = str(layer["artifact"])
        for side in ("reference", "candidate"):
            side_artifacts = _mapping(artifacts[side], f"artifacts.{side}")
            record = _mapping(side_artifacts[artifact_name], f"artifacts.{side}.{artifact_name}")
            verified, blocker = _preflight_artifact(
                root,
                side,
                artifact_name,
                record,
                required=bool(layer["required"]),
            )
            if verified is not None:
                verified_artifacts.append(verified)
            if blocker is not None:
                blockers.append(blocker)
    return verified_artifacts, blockers


def _preflight_artifact(
    root: Path,
    side: str,
    artifact_name: str,
    record: dict[str, object],
    *,
    required: bool,
) -> tuple[Record | None, Record | None]:
    path_value = record["path"]
    name = f"{side}.{artifact_name}"
    if path_value is None:
        if not required:
            return None, None
        return None, {
            "category": "artifact",
            "name": name,
            "status": "MISSING",
            "reason": "path and sha256 are not registered",
        }
    artifact_path = _resolve_under_root(root, str(path_value))
    if not artifact_path.is_file():
        return None, {
            "category": "artifact",
            "name": name,
            "status": "MISSING",
            "reason": f"file does not exist: {artifact_path}",
        }
    actual_hash = _sha256(artifact_path)
    expected_hash = str(record["sha256"]).casefold()
    hash_matches = actual_hash == expected_hash
    verified: Record = {
        "side": side,
        "artifact": artifact_name,
        "path": str(artifact_path),
        "sha256": actual_hash,
        "hash_matches": hash_matches,
    }
    if hash_matches:
        return verified, None
    return verified, {
        "category": "artifact_hash",
        "name": name,
        "status": "FAIL",
        "reason": f"expected {expected_hash}; got {actual_hash}",
    }


def _resolve_under_root(root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute():
        raise ManifestError("artifact paths must be relative to artifact_root")
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ManifestError(f"artifact path escapes artifact_root: {relative}") from error
    return resolved


def compare_matched_case(
    manifest_path: str | Path,
    artifact_root: str | Path,
) -> dict[str, object]:
    """Compare all available layers after the strict preflight passes."""

    preflight = preflight_matched_case(manifest_path, artifact_root)
    if not bool(preflight["ready_for_comparison"]):
        return {
            "tool_revision": TOOL_REVISION,
            "report_kind": "matched_case_comparison",
            "comparison_id": preflight["comparison_id"],
            "status": "BLOCKED",
            "preflight": preflight,
            "layers": {},
            "first_discrepancy_layer": "settings_or_inputs",
            "claim": _claim("insufficient_evidence"),
            "accuracy_evidence": _accuracy_summary(
                load_matched_case_manifest(manifest_path)["accuracy_evidence"]
            ),
        }
    manifest = load_matched_case_manifest(manifest_path)
    root = Path(artifact_root).expanduser().resolve()
    layers = _mapping(manifest["layers"], "layers")
    artifacts = _mapping(manifest["artifacts"], "artifacts")
    reports: dict[str, object] = {}
    for layer_name in _LAYER_ORDER:
        layer = _mapping(layers[layer_name], f"layers.{layer_name}")
        reports[layer_name] = _compare_layer(layer_name, layer, artifacts, root)
    first = _first_discrepancy(reports, layers)
    required_statuses = [
        str(_mapping(reports[name], f"report.{name}")["status"])
        for name in _LAYER_ORDER
        if bool(_mapping(layers[name], f"layers.{name}")["required"])
    ]
    if required_statuses and all(status == "PASS" for status in required_statuses):
        status = "PASS"
        supported_claim = "within_preregistered_tolerances_for_this_artifact_set"
    elif any(status == "FAIL" for status in required_statuses):
        status = "FAIL"
        supported_claim = "discrepancy_detected"
    else:
        status = "NOT_TESTED"
        supported_claim = "insufficient_evidence"
    return {
        "tool_revision": TOOL_REVISION,
        "report_kind": "matched_case_comparison",
        "comparison_id": manifest["comparison_id"],
        "status": status,
        "preflight": preflight,
        "layers": reports,
        "first_discrepancy_layer": first,
        "claim": _claim(supported_claim),
        "accuracy_evidence": _accuracy_summary(manifest["accuracy_evidence"]),
    }


def _compare_layer(
    layer_name: str,
    layer: dict[str, object],
    artifacts: dict[str, object],
    root: Path,
) -> dict[str, object]:
    artifact_name, paths = _layer_paths(layer, artifacts, root)
    if paths is None:
        return {
            "status": "NOT_TESTED",
            "reason": "optional layer artifacts are not registered",
            "artifact": artifact_name,
        }
    reference_rows, candidate_rows = _load_layer_rows(layer_name, layer, paths)
    if not reference_rows and not candidate_rows:
        return {
            "status": "NOT_TESTED",
            "reason": "both artifacts contain headers but no evidence rows",
            "artifact": artifact_name,
            "row_count": 0,
        }
    alignment_failure = _key_alignment_failure(artifact_name, reference_rows, candidate_rows)
    if alignment_failure is not None:
        return alignment_failure
    return _compare_aligned_rows(
        layer_name,
        artifact_name,
        layer,
        reference_rows,
        candidate_rows,
    )


def _layer_paths(
    layer: dict[str, object],
    artifacts: dict[str, object],
    root: Path,
) -> tuple[str, dict[str, Path] | None]:
    artifact_name = str(layer["artifact"])
    paths: dict[str, Path] = {}
    for side in ("reference", "candidate"):
        side_artifacts = _mapping(artifacts[side], f"artifacts.{side}")
        record = _mapping(side_artifacts[artifact_name], f"artifacts.{side}.{artifact_name}")
        if record["path"] is None:
            return artifact_name, None
        paths[side] = _resolve_under_root(root, str(record["path"]))
    return artifact_name, paths


def _load_layer_rows(
    layer_name: str,
    layer: dict[str, object],
    paths: dict[str, Path],
) -> tuple[
    dict[tuple[object, ...], dict[str, str]],
    dict[tuple[object, ...], dict[str, str]],
]:
    required_columns = _required_columns(layer)
    key_specs = [
        _mapping(item, f"layers.{layer_name}.keys")
        for item in _sequence(layer["keys"], f"layers.{layer_name}.keys")
    ]
    reference = _read_table(paths["reference"], required_columns["reference"])
    candidate = _read_table(paths["candidate"], required_columns["candidate"])
    return (
        _index_rows(reference, key_specs, "reference", paths["reference"]),
        _index_rows(candidate, key_specs, "candidate", paths["candidate"]),
    )


def _key_alignment_failure(
    artifact_name: str,
    reference_rows: dict[tuple[object, ...], dict[str, str]],
    candidate_rows: dict[tuple[object, ...], dict[str, str]],
) -> dict[str, object] | None:
    reference_keys = set(reference_rows)
    candidate_keys = set(candidate_rows)
    missing_candidate = sorted(reference_keys - candidate_keys)
    missing_reference = sorted(candidate_keys - reference_keys)
    if not missing_candidate and not missing_reference:
        return None
    return {
        "status": "FAIL",
        "failure_kind": "key_alignment",
        "artifact": artifact_name,
        "reference_row_count": len(reference_rows),
        "candidate_row_count": len(candidate_rows),
        "missing_candidate_count": len(missing_candidate),
        "missing_reference_count": len(missing_reference),
        "missing_candidate_examples": [list(key) for key in missing_candidate[:5]],
        "missing_reference_examples": [list(key) for key in missing_reference[:5]],
    }


def _compare_aligned_rows(
    layer_name: str,
    artifact_name: str,
    layer: dict[str, object],
    reference_rows: dict[tuple[object, ...], dict[str, str]],
    candidate_rows: dict[tuple[object, ...], dict[str, str]],
) -> dict[str, object]:
    ordered_keys = sorted(reference_rows)
    numeric_reports = [
        _numeric_metric(spec, ordered_keys, reference_rows, candidate_rows)
        for spec in (
            _mapping(item, f"layers.{layer_name}.numeric")
            for item in _sequence(layer["numeric"], f"layers.{layer_name}.numeric")
        )
    ]
    categorical_reports = [
        _categorical_metric(spec, ordered_keys, reference_rows, candidate_rows)
        for spec in (
            _mapping(item, f"layers.{layer_name}.categorical")
            for item in _sequence(layer["categorical"], f"layers.{layer_name}.categorical")
        )
    ]
    passed = all(bool(report["within_tolerance"]) for report in numeric_reports) and all(
        bool(report["exact_match"]) for report in categorical_reports
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "failure_kind": None if passed else layer_name,
        "artifact": artifact_name,
        "row_count": len(ordered_keys),
        "numeric": numeric_reports,
        "categorical": categorical_reports,
    }


def _required_columns(layer: dict[str, object]) -> dict[str, set[str]]:
    result = {"reference": set(), "candidate": set()}
    for value_spec in _sequence(layer["keys"], "layer.keys"):
        spec = _mapping(value_spec, "key spec")
        result["reference"].add(str(spec["reference_column"]))
        result["candidate"].add(str(spec["candidate_column"]))
    for group in ("numeric", "categorical"):
        for value_spec in _sequence(layer[group], f"layer.{group}"):
            spec = _mapping(value_spec, f"{group} spec")
            result["reference"].add(str(spec["reference_column"]))
            result["candidate"].add(str(spec["candidate_column"]))
    return result


def _read_table(path: Path, required: set[str]) -> _Table:
    try:
        with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ManifestError(f"{path}: CSV header must be nonempty and unique")
            missing = required.difference(reader.fieldnames)
            if missing:
                raise ManifestError(f"{path}: missing columns {sorted(missing)}")
            rows = tuple(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ManifestError(f"cannot read comparison artifact {path}: {error}") from error
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ManifestError(f"{path}: malformed CSV row")
    return _Table(tuple(reader.fieldnames), rows)


def _index_rows(
    table: _Table,
    specs: list[dict[str, object]],
    side: str,
    path: Path,
) -> dict[tuple[object, ...], dict[str, str]]:
    result: dict[tuple[object, ...], dict[str, str]] = {}
    for row_number, row in enumerate(table.rows, start=2):
        key = tuple(_key_value(row, spec, side, path, row_number) for spec in specs)
        if key in result:
            raise ManifestError(f"{path}: duplicate comparison key {key!r}")
        result[key] = row
    return result


def _key_value(
    row: dict[str, str],
    spec: dict[str, object],
    side: str,
    path: Path,
    row_number: int,
) -> object:
    token = row[str(spec[f"{side}_column"])].strip()
    if not token:
        raise ManifestError(f"{path}: blank key at row {row_number}")
    kind = str(spec["kind"])
    if kind == "text":
        return token
    try:
        if kind == "integer":
            value = int(token)
            if str(value) != token and token not in {f"+{value}", f"-{abs(value)}"}:
                raise ValueError
            return value
        value = float(token)
    except ValueError as error:
        raise ManifestError(f"{path}: invalid {kind} key at row {row_number}: {token!r}") from error
    if not math.isfinite(value):
        raise ManifestError(f"{path}: nonfinite float key at row {row_number}")
    return value


def _numeric_metric(
    spec: dict[str, object],
    keys: list[tuple[object, ...]],
    reference_rows: dict[tuple[object, ...], dict[str, str]],
    candidate_rows: dict[tuple[object, ...], dict[str, str]],
) -> dict[str, object]:
    reference = _numeric_values(keys, reference_rows, str(spec["reference_column"]), "reference")
    candidate = _numeric_values(keys, candidate_rows, str(spec["candidate_column"]), "candidate")
    difference = np.abs(candidate - reference)
    absolute = _nonnegative_float(spec["absolute_tolerance"], "absolute_tolerance")
    relative = _nonnegative_float(spec["relative_tolerance"], "relative_tolerance")
    scale = _nonnegative_float(spec["scale"], "scale")
    limit = absolute + relative * np.maximum(np.abs(reference), scale)
    within = difference <= limit
    zero_limit_failures = int(np.count_nonzero((limit == 0.0) & (difference != 0.0)))
    positive = limit > 0.0
    maximum_ratio: float | None
    if zero_limit_failures:
        maximum_ratio = None
    elif bool(positive.any()):
        maximum_ratio = float(np.max(difference[positive] / limit[positive]))
    else:
        maximum_ratio = 0.0
    failed_indices = np.flatnonzero(~within)
    return {
        "name": spec["name"],
        "unit": spec["unit"],
        "count": len(keys),
        "absolute_tolerance": absolute,
        "relative_tolerance": relative,
        "scale": scale,
        "within_tolerance": bool(within.all()),
        "failure_count": int(failed_indices.size),
        "zero_limit_failure_count": zero_limit_failures,
        "absolute_error_p50": float(np.percentile(difference, 50.0)),
        "absolute_error_p90": float(np.percentile(difference, 90.0)),
        "absolute_error_p99": float(np.percentile(difference, 99.0)),
        "absolute_error_max": float(np.max(difference)),
        "maximum_tolerance_ratio": maximum_ratio,
        "first_failure_key": None
        if failed_indices.size == 0
        else list(keys[int(failed_indices[0])]),
    }


def _numeric_values(
    keys: list[tuple[object, ...]],
    rows: dict[tuple[object, ...], dict[str, str]],
    column: str,
    side: str,
) -> np.ndarray:
    result = np.empty(len(keys), dtype=np.float64)
    for index, key in enumerate(keys):
        token = rows[key][column].strip()
        try:
            value = float(token)
        except ValueError as error:
            raise ManifestError(f"{side}.{column}: nonnumeric value at key {key!r}") from error
        if not math.isfinite(value):
            raise ManifestError(f"{side}.{column}: nonfinite value at key {key!r}")
        result[index] = value
    return result


def _categorical_metric(
    spec: dict[str, object],
    keys: list[tuple[object, ...]],
    reference_rows: dict[tuple[object, ...], dict[str, str]],
    candidate_rows: dict[tuple[object, ...], dict[str, str]],
) -> dict[str, object]:
    reference_column = str(spec["reference_column"])
    candidate_column = str(spec["candidate_column"])
    mismatches = [
        key
        for key in keys
        if reference_rows[key][reference_column] != candidate_rows[key][candidate_column]
    ]
    return {
        "name": spec["name"],
        "count": len(keys),
        "exact_match": not mismatches,
        "mismatch_count": len(mismatches),
        "first_mismatch_key": None if not mismatches else list(mismatches[0]),
    }


def _first_discrepancy(
    reports: dict[str, object],
    layers: dict[str, object],
) -> str:
    owner = {
        "field": "field_or_geometry_sampling",
        "force": "force_or_charge_rhs",
        "integrator": "integrator_or_stage_coupling",
        "trajectory": "trajectory_accumulation_after_probes",
        "boundary": "boundary_event_or_wall_mapping",
    }
    for name in _LAYER_ORDER:
        report = _mapping(reports[name], f"report.{name}")
        if report["status"] == "FAIL":
            return owner[name]
        layer = _mapping(layers[name], f"layers.{name}")
        if bool(layer["required"]) and report["status"] == "NOT_TESTED":
            return f"insufficient_evidence_{name}"
    return "none_within_registered_tolerances"


def _accuracy_summary(value: object) -> dict[str, object]:
    rows = [
        dict(_mapping(item, "accuracy evidence")) for item in _sequence(value, "accuracy_evidence")
    ]
    confirmed = bool(rows) and all(row["status"] in {"CONFIRMED", "NOT_APPLICABLE"} for row in rows)
    return {
        "status": "AVAILABLE_NOT_EQUIVALENCE_PROOF" if confirmed else "NOT_TESTED",
        "items": rows,
        "interpretation": (
            "Convergence and uncertainty evidence can support the registered tolerance budget; "
            "it cannot make COMSOL a golden truth or establish equal solver accuracy."
        ),
    }


def _claim(supported_claim: str) -> dict[str, str]:
    return {
        "supported_claim": supported_claim,
        "comsol_equal_accuracy": "NOT_CLAIMED",
        "golden_truth": "NOT_CLAIMED",
        "interpretation": (
            "PASS means only within preregistered tolerances for the hashed artifacts and scope."
        ),
    }


def write_report(path: str | Path, report: dict[str, object]) -> None:
    """Write one JSON report without replacing existing evidence."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", errors="strict") as stream:
        stream.write(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "compare"):
        child = subparsers.add_parser(command)
        child.add_argument("manifest", type=Path)
        child.add_argument("--artifact-root", required=True, type=Path)
        child.add_argument("--output", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "preflight":
        report = preflight_matched_case(args.manifest, args.artifact_root)
    else:
        report = compare_matched_case(args.manifest, args.artifact_root)
    write_report(args.output, report)
    status = str(report["status"])
    if status == "PASS":
        return 0
    if status in {"BLOCKED", "NOT_TESTED"}:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
