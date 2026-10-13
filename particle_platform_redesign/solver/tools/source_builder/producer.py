"""Realized source conversion; no expression evaluation or distribution sampling."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray

from chamber_particles import load_case
from chamber_particles.case_format import DataBundle, RealizedTableSource, write
from chamber_particles.yaml_input import parse_document

type FloatArray = NDArray[np.float64]
type VelocityMode = Literal["vector", "speed_direction", "momentum", "energy_direction"]

REVISION = "realized_initial_rows_v1"
_ELEMENTARY_CHARGE_C = 1.602176634e-19
_LENGTH_UNITS = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}
_FIELD_UNITS = {
    "release_time_s": {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9},
    "position_m": _LENGTH_UNITS,
    "mass_kg": {"kg": 1.0, "g": 1e-3, "mg": 1e-6, "ug": 1e-9, "ng": 1e-12},
    "drag_diameter_m": _LENGTH_UNITS,
    "contact_radius_m": _LENGTH_UNITS,
    "electrostatic_radius_m": _LENGTH_UNITS,
    "displaced_volume_m3": {"m3": 1.0, "cm3": 1e-6, "mm3": 1e-9, "um3": 1e-18},
    "charge_number": {"1": 1.0},
    "model_weight": {"1": 1.0},
}
_PROPERTIES = (
    "charge_number",
    "mass_kg",
    "drag_diameter_m",
    "contact_radius_m",
    "electrostatic_radius_m",
    "displaced_volume_m3",
    "model_weight",
    "material_id",
)


def _finite(values: ArrayLike, label: str) -> FloatArray:
    array = np.asarray(values, dtype="<f8")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    return array


def _retained_components(
    values: ArrayLike, components: Sequence[str], coordinate_system: str
) -> FloatArray:
    array = _finite(values, "vector components")
    names = tuple(components)
    orders = {
        "cartesian_xy": (("x", "y"), ("x", "y", "z"), 2),
        "axisymmetric_rz": (("r", "z"), ("r", "phi", "z"), 1),
    }
    if coordinate_system not in orders:
        raise ValueError("only cartesian_xy and axisymmetric_rz are supported")
    pair, triple, third = orders[coordinate_system]
    if array.ndim != 2 or array.shape[1] != len(names):
        raise ValueError("vector shape must match the declared component order")
    if names == pair:
        return array
    if names != triple:
        raise ValueError("component order must be the canonical pair or declared 3-component order")
    if np.any(array[:, third] != 0.0):
        raise ValueError("a nonzero third component is unsupported; projection is forbidden")
    return np.delete(array, third, axis=1)


def _rotate(vector: FloatArray, basis_to_case: ArrayLike) -> FloatArray:
    basis = _finite(basis_to_case, "basis_to_case")
    if basis.shape not in ((2, 2), (vector.shape[0], 2, 2)):
        raise ValueError("basis_to_case must have shape [2,2] or [N,2,2]")
    gram = np.swapaxes(basis, -1, -2) @ basis
    budget = 64 * np.finfo(np.float64).eps
    if not np.allclose(gram, np.eye(2), rtol=0, atol=budget) or not np.allclose(
        np.linalg.det(basis), 1.0, rtol=0, atol=budget
    ):
        raise ValueError("basis_to_case must be an oriented orthonormal basis")
    return np.asarray(np.einsum("...ij,...j->...i", basis, vector), dtype="<f8")


def _direction_vectors(
    direction: ArrayLike | None, count: int, components: Sequence[str], coordinate_system: str
) -> FloatArray:
    if direction is None:
        raise ValueError("a direction vector is required")
    vector = _retained_components(direction, components, coordinate_system)
    if vector.shape[0] != count:
        raise ValueError("direction rows must match the particle rows")
    # Scale first to normalize extreme finite vectors without overflow/underflow.
    scale = np.max(np.abs(vector), axis=1)
    if np.any(scale == 0):
        raise ValueError("zero direction is invalid")
    scaled = vector / scale[:, None]
    return scaled / np.linalg.norm(scaled, axis=1)[:, None]


def convert_velocity(
    values: ArrayLike,
    mass_kg: ArrayLike,
    *,
    mode: VelocityMode,
    unit: str,
    components: Sequence[str],
    coordinate_system: str,
    basis_to_case: ArrayLike,
    frame: str,
    direction: ArrayLike | None = None,
) -> FloatArray:
    """Convert row-correlated nonrelativistic inertial initial data to SI vectors.

    Positions already use the canonical case basis. ``basis_to_case`` has source
    vector basis columns in that basis; it may be fixed or supplied per row.
    ``frame='inertial'`` is a caller declaration, not a native-frame inference.
    """
    if frame != "inertial":
        raise ValueError("moving/rotating or unknown frames require external physical resolution")
    mass = _finite(mass_kg, "mass_kg")
    if mass.ndim != 1 or np.any(mass <= 0):
        raise ValueError("mass_kg must be a positive row vector")
    raw = _finite(values, mode)
    if mode in ("vector", "momentum"):
        vector = _component_velocity(raw, mass, mode, unit, components, coordinate_system)
        if direction is not None:
            raise ValueError("vector/momentum input does not use a separate direction")
    elif mode in ("speed_direction", "energy_direction"):
        speed = _scalar_speed(raw, mass, mode, unit)
        vector = speed[:, None] * _direction_vectors(
            direction, mass.size, components, coordinate_system
        )
    else:
        raise ValueError(f"unsupported velocity mode: {mode}")
    return _finite(_rotate(vector, basis_to_case), "converted velocity_m_s")


def _component_velocity(
    raw: FloatArray,
    mass: FloatArray,
    mode: str,
    unit: str,
    components: Sequence[str],
    coordinate_system: str,
) -> FloatArray:
    vector = _retained_components(raw, components, coordinate_system)
    if vector.shape[0] != mass.size:
        raise ValueError("velocity/momentum rows must match mass_kg")
    units = (
        {"m/s": 1.0, "cm/s": 1e-2, "km/s": 1e3}
        if mode == "vector"
        else {
            "kg*m/s": 1.0,
            "g*cm/s": 1e-5,
        }
    )
    vector = vector * _unit_factor(unit, units, mode)
    return vector if mode == "vector" else vector / mass[:, None]


def _scalar_speed(raw: FloatArray, mass: FloatArray, mode: str, unit: str) -> FloatArray:
    if raw.shape != mass.shape or np.any(raw < 0):
        raise ValueError("speed/energy must be nonnegative with one value per particle")
    if mode == "speed_direction":
        return raw * _unit_factor(unit, {"m/s": 1.0, "cm/s": 1e-2, "km/s": 1e3}, mode)
    energy = raw * _unit_factor(
        unit, {"J": 1.0, "eV": _ELEMENTARY_CHARGE_C, "keV": 1e3 * _ELEMENTARY_CHARGE_C}, mode
    )
    return np.sqrt(2.0 * (energy / mass))


def _unit_factor(unit: str, supported: Mapping[str, float], label: str) -> float:
    if unit not in supported:
        raise ValueError(f"unsupported or missing unit for {label}: {unit!r}")
    return supported[unit]


def _integer_column(rows: list[dict[str, str]], column: str, dtype: str) -> NDArray:
    values = [Decimal(row[column]) for row in rows]
    if any(not value.is_finite() or value != value.to_integral_value() for value in values):
        raise ValueError(f"{column} must contain exact integers")
    bounds = np.iinfo(np.dtype(dtype))
    if any(value < int(bounds.min) or value > int(bounds.max) for value in values):
        raise ValueError(f"{column} is outside the canonical integer range")
    return np.asarray([int(value) for value in values], dtype=dtype)


def _csv_values(rows: list[dict[str, str]], columns: str | Sequence[str]) -> FloatArray:
    if isinstance(columns, str):
        return _finite([float(row[columns]) for row in rows], columns)
    return _finite([[float(row[name]) for name in columns] for row in rows], "CSV components")


def _select_rows(raw: bytes, filters: Mapping[str, str | float]) -> list[dict[str, str]]:
    with io.StringIO(raw.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV headers must be present and unique")
        selected = [row for row in reader if _matches(row, filters)]
    if not selected:
        raise ValueError("CSV selection contains no initial rows")
    return selected


def _matches(row: dict[str, str], filters: Mapping[str, str | float]) -> bool:
    for name, expected in filters.items():
        value = row[name]
        if isinstance(expected, str):
            equal = value == expected
        else:
            equal = float(value) == expected
        if not equal:
            return False
    return True


def _configuration(config: Mapping[str, Any]) -> None:
    required = {
        "format_version",
        "table",
        "columns",
        "units",
        "velocity_mode",
        "velocity_columns",
        "velocity_unit",
        "components",
        "basis_to_case",
        "frame",
        "metadata",
    }
    optional = {"direction_columns", "filter_equals"}
    if set(config) - required - optional or required - set(config):
        raise ValueError("source configuration has missing or unknown keys")
    if config["format_version"] != 1:
        raise ValueError("only source configuration format_version 1 is supported")
    columns = config["columns"]
    if set(columns) - {"particle_id", "release_time_s", "position_m", *_PROPERTIES}:
        raise ValueError("unknown canonical source column")
    if {"particle_id", "release_time_s", "position_m"} - set(columns):
        raise ValueError("CSV must explicitly supply particle ID, release time and position")
    if set(config["units"]) - set(columns):
        raise ValueError("units must name supplied columns")
    for name in {"particle_id", "material_id"} & set(config["units"]):
        if config["units"][name] != "1":
            raise ValueError(f"{name} is dimensionless")
    _source_metadata(config["metadata"])


def _source_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("metadata must explicitly record source scope and unresolved assumptions")
    if not isinstance(metadata.get("scope"), str) or not metadata["scope"]:
        raise ValueError("metadata.scope must state the source preparation claim")
    if not isinstance(metadata.get("unresolved"), list) or any(
        not isinstance(item, str) for item in metadata["unresolved"]
    ):
        raise ValueError("metadata.unresolved must explicitly list unresolved assumptions")


def _property_values(
    name: str,
    rows: list[dict[str, str]],
    config: Mapping[str, Any],
    base: RealizedTableSource,
    index: NDArray[np.int64],
) -> NDArray:
    columns = config["columns"]
    if name not in columns:
        if np.any(index < 0):
            raise ValueError(f"new particle IDs require explicit {name}; no default is supplied")
        return np.array(getattr(base, name)[index], copy=True)
    if name == "material_id":
        return _integer_column(rows, columns[name], "<i4")
    units = _FIELD_UNITS[name]
    unit = config["units"].get(name, "1" if name in ("charge_number", "model_weight") else "")
    return _csv_values(rows, columns[name]) * _unit_factor(unit, units, name)


def read_csv_source(
    path: str | Path, data: DataBundle, config: Mapping[str, Any]
) -> RealizedTableSource:
    """Read realized rows, inheriting only missing properties of exact existing IDs.

    A new population can reuse the template geometry/fields when every property
    is supplied. This function returns rows; the canonical writer owns schema
    validation. It never evaluates native expressions or generates random rows.
    """
    raw = Path(path).read_bytes()
    return _source_from_rows(_select_rows(raw, config.get("filter_equals", {})), data, config)


def _source_from_rows(
    rows: list[dict[str, str]], data: DataBundle, config: Mapping[str, Any]
) -> RealizedTableSource:
    _configuration(config)
    tables = {source.name: source for source in data.sources}
    base = tables[config["table"]]
    if not isinstance(base, RealizedTableSource):
        raise ValueError("this producer accepts internal table sources only")
    columns = config["columns"]
    ids = _integer_column(rows, columns["particle_id"], "<i8")
    if np.unique(ids).size != ids.size:
        raise ValueError("selected initial rows have duplicate particle IDs")
    lookup = {int(value): i for i, value in enumerate(base.particle_id)}
    index = np.asarray([lookup.get(int(value), -1) for value in ids], dtype="<i8")
    properties = {name: _property_values(name, rows, config, base, index) for name in _PROPERTIES}
    position = _csv_values(rows, columns["position_m"]) * _unit_factor(
        config["units"].get("position_m", ""), _LENGTH_UNITS, "position_m"
    )
    release = _csv_values(rows, columns["release_time_s"]) * _unit_factor(
        config["units"].get("release_time_s", ""), _FIELD_UNITS["release_time_s"], "release_time_s"
    )
    velocity = _csv_velocity(rows, properties["mass_kg"], data.coordinate_system, config)
    return RealizedTableSource(
        name=base.name,
        particle_id=ids,
        release_time_s=release,
        position_m=position,
        velocity_m_s=velocity,
        **properties,
    )


def _csv_velocity(
    rows: list[dict[str, str]], mass: FloatArray, coordinate_system: str, config: Mapping[str, Any]
) -> FloatArray:
    mode = config["velocity_mode"]
    columns = config["velocity_columns"]
    if mode in ("speed_direction", "energy_direction"):
        if len(columns) != 1:
            raise ValueError("speed/energy requires exactly one CSV value column")
        columns = columns[0]
    direction_columns = config.get("direction_columns")
    direction = None if direction_columns is None else _csv_values(rows, direction_columns)
    return convert_velocity(
        _csv_values(rows, columns),
        mass,
        mode=mode,
        unit=config["velocity_unit"],
        components=config["components"],
        coordinate_system=coordinate_system,
        basis_to_case=config["basis_to_case"],
        frame=config["frame"],
        direction=direction,
    )


def prepare_case(
    template: str | Path, rows: str | Path, configuration: str | Path, output: str | Path
) -> dict[str, str]:
    """Write a fresh canonical case through the sole writer, then publicly load it."""
    template_path, rows_path, config_path = map(Path, (template, rows, configuration))
    template_raw, rows_raw, config_raw = (
        template_path.read_bytes(),
        rows_path.read_bytes(),
        config_path.read_bytes(),
    )
    parsed_config = parse_document(config_raw)
    if not isinstance(parsed_config, dict):
        raise ValueError("source configuration must be a mapping")
    config = cast(dict[str, Any], parsed_config)
    _configuration(config)
    case = load_case(template_path)
    if template_path.read_bytes() != template_raw:
        raise ValueError("template changed during canonical loading")
    source = _source_from_rows(
        _select_rows(rows_raw, config.get("filter_equals", {})), case.data, config
    )
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    data_path, yaml_path = destination / "case.h5", destination / "case.yaml"
    if data_path.exists() or yaml_path.exists():
        raise FileExistsError("source builder refuses to replace an existing case")
    snapshots = {
        "template_yaml": template_raw,
        "rows_csv": rows_raw,
        "configuration_json": config_raw,
    }
    inputs = {name: "sha256:" + hashlib.sha256(raw).hexdigest() for name, raw in snapshots.items()}
    inputs["template_content_hash"] = case.content_hash
    provenance = _provenance(case.data, inputs, config)
    data = replace(
        case.data,
        provenance_json=provenance,
        sources=tuple(source if item.name == source.name else item for item in case.data.sources),
    )
    info = write(data_path, data)
    document = cast(dict[str, Any], parse_document(template_raw))
    document["case"]["data_path"] = "case.h5"
    document["case"]["expected_content_hash"] = info.content_hash
    with yaml_path.open("x", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(document, stream, sort_keys=False)
    load_case(yaml_path)
    return {"case": str(yaml_path), "data": str(data_path), "content_hash": info.content_hash}


def _provenance(data: DataBundle, inputs: dict[str, str], config: Mapping[str, Any]) -> str:
    upstream = json.loads(data.provenance_json)
    source_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return json.dumps(
        {
            "producer": "source-builder",
            "producer_version": REVISION,
            "source_sha256": "sha256:" + source_hash,
            "field_semantics_revision": upstream["field_semantics_revision"],
            "producer_metadata": {
                "upstream": upstream,
                "source_builder": {
                    "revision": REVISION,
                    "inputs": inputs,
                    "configuration": dict(config),
                    "inheritance": "missing properties joined by exact particle_id only",
                    "sampling": "none; supplied joint realization retained",
                },
            },
        }
    )
