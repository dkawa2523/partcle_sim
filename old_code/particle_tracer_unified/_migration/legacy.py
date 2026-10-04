"""Legacy value normalization and retired-source detection."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


class RemovedSourceGenerationError(ValueError):
    def __init__(self, findings: Sequence[str]):
        self.findings = tuple(str(item) for item in findings)
        preview = "; ".join(self.findings)
        super().__init__(
            "removed source-generation behavior cannot be migrated automatically: "
            + preview
        )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _legacy_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
        raise ValueError(f"ambiguous legacy boolean value {value!r}")
    return bool(value)


def _token(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _canonical_choice(
    value: Any,
    *,
    canonical: Sequence[str],
    aliases: Mapping[str, str],
    label: str,
) -> str:
    choices = {_token(item): str(item) for item in canonical}
    choices.update({_token(alias): target for alias, target in aliases.items()})
    token = _token(value)
    if token not in choices:
        expected = sorted(set(choices.values()))
        raise ValueError(
            f"unsupported legacy {label} {value!r}; expected one of {expected}"
        )
    return choices[token]


def _canonical_keys(
    value: Mapping[str, Any],
    *,
    canonical: Sequence[str],
    aliases: Mapping[str, str],
    label: str,
) -> dict[str, Any]:
    choices = {_token(item): str(item) for item in canonical}
    choices.update({_token(alias): target for alias, target in aliases.items()})
    result: dict[str, Any] = {}
    for raw_name, item in value.items():
        token = _token(raw_name)
        if token not in choices:
            raise ValueError(f"unknown legacy {label} key {raw_name!r}")
        name = choices[token]
        if name in result:
            raise ValueError(
                f"legacy {label} supplies {name!r} more than once through aliases"
            )
        result[name] = item
    return result


def _merge_without_conflicts(
    target: dict[str, Any],
    incoming: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for name, value in incoming.items():
        if name in target and target[name] != value:
            raise ValueError(f"legacy {label} supplies conflicting values for {name!r}")
        target.setdefault(name, value)


def _resolve(base: Path, value: Any, *, label: str) -> Path:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"legacy config is missing {label}")
    path = Path(text)
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {resolved}")
    return resolved


def _relocated_reference(source_base: Path, destination_base: Path, value: Any) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError("legacy path reference must not be blank")
    path = Path(text)
    resolved = path.resolve() if path.is_absolute() else (source_base / path).resolve()
    return Path(os.path.relpath(resolved, destination_base)).as_posix()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"could not read legacy config {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("legacy YAML root must be a mapping")
    if "case" in value or int(value.get("schema_version", 0) or 0) == 2:
        raise ValueError(
            "configuration is already v0.2; migration accepts only legacy run YAML"
        )
    return dict(value)


def _active_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower()


def _configured_source_path_findings(config: Mapping[str, Any]) -> list[str]:
    paths = _mapping(config.get("paths"))
    names = ("source_events_csv", "process_steps_csv", "recipe_manifest_yaml")
    return [
        f"paths.{name}"
        for name in names
        if paths.get(name) is not None and str(paths.get(name)).strip()
    ]


def _configured_source_policy_findings(config: Mapping[str, Any]) -> list[str]:
    findings: list[str] = []
    source = _mapping(config.get("source"))
    for name in ("law", "default_law"):
        law = _active_text(source.get(name))
        if law and law != "explicit_csv":
            findings.append(f"source.{name}={law}")

    preprocess = _mapping(source.get("preprocess"))
    if _legacy_bool(preprocess.get("boundary_release", False), default=False):
        findings.append("source.preprocess.boundary_release")
    policy = _active_text(preprocess.get("normal_velocity_policy"))
    if policy and policy != "keep":
        findings.append(f"source.preprocess.normal_velocity_policy={policy}")
    return findings


def _table_source_law_findings(
    table_name: str,
    frame: pd.DataFrame,
    column: str,
) -> list[str]:
    if column not in frame:
        return []
    findings: list[str] = []
    for row_index, raw_value in enumerate(frame[column].tolist(), start=2):
        value = _active_text(raw_value)
        if value and value != "explicit_csv":
            findings.append(f"{table_name}:row {row_index}:{column}={value}")
    return findings


def _table_source_generation_findings(
    table_name: str,
    frame: pd.DataFrame,
) -> list[str]:
    findings: list[str] = []
    for column in ("source_law", "source_law_default", "source_law_override"):
        findings.extend(_table_source_law_findings(table_name, frame, column))
    if "source_event_tag" not in frame:
        return findings
    for row_index, tag_raw in enumerate(frame["source_event_tag"].tolist(), start=2):
        if _active_text(tag_raw):
            findings.append(f"{table_name}:row {row_index}:source_event_tag")
    return findings


def _source_generation_findings(
    config: Mapping[str, Any], tables: Sequence[tuple[str, pd.DataFrame]]
) -> list[str]:
    findings = _configured_source_path_findings(config)
    findings.extend(_configured_source_policy_findings(config))
    for table_name, frame in tables:
        findings.extend(_table_source_generation_findings(table_name, frame))
    return findings
