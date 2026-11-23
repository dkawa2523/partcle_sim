"""Bind explicitly generated field groups at normal input publication.

Only declared dependencies are checked. This module never infers dependencies
from model/field names and never selects or regenerates a physical model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from chamber_particles.case_format import DataBundle, field_identity

GENERATION_REVISION = "declared_field_generation_v1"
_PROPERTY_NAMES = {"mass_kg", "drag_diameter_m", "electrostatic_radius_m", "displaced_volume_m3"}


def _parameter_hash(document: Mapping[str, object]) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _bindings(data: DataBundle, names: Sequence[str]) -> dict[str, str]:
    fields = {item.name: item for item in data.fields}
    layouts = {item.name: item for item in data.layouts}
    if not names or len(set(names)) != len(names):
        raise ValueError("declared generation requires a nonempty unique field group")
    result = {}
    for name in names:
        if name not in fields:
            raise ValueError(f"declared generated group is incomplete: missing field {name!r}")
        item = fields[name]
        if item.layout not in layouts:
            raise ValueError(f"declared field {name!r} has no layout {item.layout!r}")
        result[name] = field_identity(item, layouts[item.layout], data.coordinate_system)
    return result


def _properties(data: DataBundle, expected: Mapping[str, object]) -> None:
    if not expected or not set(expected).issubset(_PROPERTY_NAMES) or not data.sources:
        raise ValueError("declared generation requires supported homogeneous particle properties")
    for name, value in expected.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not np.isfinite(value):
            raise ValueError(f"declared particle property {name!r} must be finite")
        for source in data.sources:
            values = getattr(source, name)
            if not values.size or not np.all(values == value):
                raise ValueError(
                    f"stale generated fields: homogeneous particle property {name!r} changed"
                )


def record_generation(
    data: DataBundle,
    *,
    primitive_names: Sequence[str],
    parameter_document: Mapping[str, object],
    particle_properties: Mapping[str, object],
    output_names: Sequence[str],
    producer_revision: str,
) -> dict[str, object]:
    """Record one producer's complete group after deriving its field values."""
    _properties(data, particle_properties)
    if not producer_revision:
        raise ValueError("declared generation requires its producer revision")
    return {
        "revision": GENERATION_REVISION,
        "producer_revision": producer_revision,
        "primitives": _bindings(data, primitive_names),
        "parameters": dict(parameter_document),
        "parameters_sha256": _parameter_hash(parameter_document),
        "particle_properties": dict(particle_properties),
        "outputs": _bindings(data, output_names),
    }


def _validate_record(data: DataBundle, record: Any) -> None:
    keys = {
        "revision",
        "producer_revision",
        "primitives",
        "parameters",
        "parameters_sha256",
        "particle_properties",
        "outputs",
    }
    if not isinstance(record, dict) or set(record) != keys:
        raise ValueError("declared field generation has an invalid record shape")
    if record["revision"] != GENERATION_REVISION or not isinstance(
        record["producer_revision"], str
    ):
        raise ValueError("unsupported declared field generation revision")
    if not record["producer_revision"]:
        raise ValueError("declared generation requires its producer revision")
    for name in ("parameters", "particle_properties", "primitives", "outputs"):
        if not isinstance(record[name], dict):
            raise ValueError(f"declared generation {name} must be a mapping")
    if _parameter_hash(record["parameters"]) != record["parameters_sha256"]:
        raise ValueError("stale generated fields: parameter document changed")
    _properties(data, record["particle_properties"])
    for name in ("primitives", "outputs"):
        declared = record[name]
        if any(not isinstance(key, str) for key in declared):
            raise ValueError("declared field group names must be strings")
        if _bindings(data, tuple(declared)) != declared:
            raise ValueError(f"stale generated fields: {name} field/layout/clock/basis changed")


def validate_declared_generations(data: DataBundle) -> None:
    """Reject stale/partial declared groups; independent fields remain legal."""
    provenance = json.loads(data.provenance_json)
    if not isinstance(provenance, dict):
        raise ValueError("canonical provenance must be an object")
    records = provenance.get("derived_field_generations", [])
    if not isinstance(records, list):
        raise ValueError("derived_field_generations must be a list")
    outputs: set[str] = set()
    for record in records:
        _validate_record(data, record)
        names = set(record["outputs"])
        if outputs.intersection(names):
            raise ValueError("a derived field cannot belong to multiple generation groups")
        outputs.update(names)
