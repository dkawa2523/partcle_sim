"""Publish checked runnable cases from canonical facts and explicit settings."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import yaml

from chamber_particles import check_case, load_case
from chamber_particles.case import SimulationCase
from chamber_particles.case_format import DataBundle, RealizedSource, read_with_info, write
from chamber_particles.yaml_input import parse_document


def create_case(
    bundle: DataBundle,
    settings: Mapping[str, object],
    output_directory: str | Path,
    *,
    supporting_files: Mapping[str, bytes] | None = None,
) -> SimulationCase:
    """Publish one checked case in a new or empty directory.

    ``settings`` is the normal case document. Only its data path and content
    hash are supplied here. The bundle owns available source tables; settings
    explicitly select releases. Supporting editable inputs are published in the
    same directory commit. Scientific validation belongs to the canonical reader
    and the execution preflight.
    """
    destination = Path(output_directory).expanduser().resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"case publication requires a new or empty directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = dict(settings)
    reference = document.get("case")
    if not isinstance(reference, Mapping):
        raise ValueError("settings.case must contain the case name")
    document["case"] = dict(reference)
    with TemporaryDirectory(prefix=".case-publication-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "case"
        staging.mkdir()
        info = write(staging / "case.h5", bundle)
        document["case"].update(data_path="case.h5", expected_content_hash=info.content_hash)
        (staging / "case.yaml").write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
        for name, content in (supporting_files or {}).items():
            relative = Path(name)
            if (
                relative.is_absolute()
                or relative.drive
                or ".." in relative.parts
                or relative.name in {"", "."}
                or relative.parts[0] in {"case.h5", "case.yaml"}
            ):
                raise ValueError(f"invalid supporting file path: {name!r}")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(content)
        check_case(load_case(staging / "case.yaml"))
        if destination.exists():
            destination.rmdir()  # Only an empty caller-owned destination is accepted.
        staging.rename(destination)
    return load_case(destination / "case.yaml")


def create_from_configuration(
    recipe_path: str | Path, output_directory: str | Path
) -> SimulationCase:
    """Create from a format-2 recipe declaring the complete source-table set.

    Relative paths belong to the recipe directory. Each source entry has
    ``csv`` and ``configuration`` (inline mapping or a configuration path), or
    ``existing_table`` naming an input-bundle table. Unlisted tables are removed.
    Normal settings select which of the resulting tables actually release.
    """
    path = Path(recipe_path).expanduser().resolve()
    recipe_raw = path.read_bytes()
    recipe = parse_document(recipe_raw)
    if not isinstance(recipe, dict) or set(recipe) != {
        "format_version",
        "input_bundle",
        "sources",
        "settings",
    }:
        raise ValueError("case recipe requires format_version, input_bundle, sources and settings")
    if type(recipe["format_version"]) is not int or recipe["format_version"] != 2:
        raise ValueError("only case recipe format_version 2 is supported")
    if not isinstance(recipe["settings"], dict) or not isinstance(recipe["sources"], list):
        raise ValueError("recipe settings must be a mapping and sources must be a list")
    bundle, info = read_with_info(_relative_path(recipe["input_bundle"], path.parent))
    sources: list[RealizedSource] = []
    requested: set[str] = set()
    source_receipts: list[dict[str, object]] = []
    for raw in recipe["sources"]:
        source, receipt = _recipe_source(raw, bundle, path.parent)
        if source.name in requested:
            raise ValueError(f"duplicate recipe source table: {source.name}")
        requested.add(source.name)
        sources.append(source)
        source_receipts.append(receipt)
    upstream = json.loads(bundle.provenance_json)
    provenance = {
        "producer": "chamber_particles_tools.inputs",
        "producer_version": "case_recipe_v2",
        "source_sha256": "sha256:" + hashlib.sha256(recipe_raw).hexdigest(),
        "field_semantics_revision": upstream["field_semantics_revision"],
        "producer_metadata": {
            "upstream": upstream,
            "input_content_hash": info.content_hash,
            "sources": source_receipts,
        },
    }
    bundle = replace(bundle, provenance_json=json.dumps(provenance), sources=tuple(sources))
    return create_case(bundle, recipe["settings"], output_directory)


def _relative_path(value: object, directory: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("input paths must be nonempty strings")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else directory / path).resolve()


def _recipe_source(
    raw: object, bundle: DataBundle, directory: Path
) -> tuple[RealizedSource, dict[str, object]]:
    from .source_builder import read_csv_source

    if isinstance(raw, dict) and set(raw) == {"existing_table"}:
        name = raw["existing_table"]
        if not isinstance(name, str) or not name:
            raise ValueError("existing_table must name an input-bundle source table")
        source = next((source for source in bundle.sources if source.name == name), None)
        if source is None:
            raise ValueError(f"input bundle has no source table {name!r}")
        return source, {"table": name, "origin": "input_bundle"}
    if not isinstance(raw, dict) or set(raw) != {"csv", "configuration"}:
        raise ValueError("each source requires existing_table, or csv and configuration")
    configuration = raw["configuration"]
    if isinstance(configuration, str):
        configuration_raw = _relative_path(configuration, directory).read_bytes()
        configuration = parse_document(configuration_raw)
    else:
        configuration_raw = yaml.safe_dump(configuration, sort_keys=True).encode("utf-8")
    if not isinstance(configuration, Mapping):
        raise ValueError("source configuration must be a mapping or configuration path")
    csv_path = _relative_path(raw["csv"], directory)
    csv_raw = csv_path.read_bytes()
    source = read_csv_source(csv_path, bundle, cast(Mapping[str, Any], configuration))
    if csv_path.read_bytes() != csv_raw:
        raise ValueError("source CSV changed during preparation")
    receipt: dict[str, object] = {
        "table": source.name,
        "configuration": dict(configuration),
        "csv_sha256": "sha256:" + hashlib.sha256(csv_raw).hexdigest(),
        "configuration_sha256": "sha256:" + hashlib.sha256(configuration_raw).hexdigest(),
    }
    return source, receipt


def init_example(name: str, output_directory: str | Path) -> SimulationCase:
    """Create one small editable example without external software or datasets."""
    from .examples import initialize

    return initialize(name, output_directory)
