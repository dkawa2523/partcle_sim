"""Realized source conversion; no expression evaluation or distribution sampling."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chamber_particles.case_format import (
    DataBundle,
    RealizedSource,
    RealizedSurfaceSource,
    RealizedTableSource,
)

type FloatArray = NDArray[np.float64]
type VelocityMode = Literal["vector", "speed_direction", "momentum", "energy_direction"]
type SourceKind = Literal["table", "surface"]
type SourceFilter = tuple[str, str | Decimal]


@dataclass(frozen=True, slots=True)
class _SourceColumns:
    particle_id: str
    release_time: str
    location: tuple[str, ...]
    properties: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _CSVVelocity:
    mode: VelocityMode
    columns: tuple[str, ...]
    unit: str
    components: tuple[str, ...]
    basis_to_case: FloatArray
    frame: str
    direction_columns: tuple[str, ...] | None


@dataclass(frozen=True, slots=True)
class _SourceSpecification:
    table: str
    kind: SourceKind
    columns: _SourceColumns
    unit_factors: Mapping[str, float]
    velocity: _CSVVelocity
    filters: tuple[SourceFilter, ...]
    inherit_table: str | None


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
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = np.asarray(values, dtype="<f8")
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{label} must contain representable numeric values") from error
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
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
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
    try:
        values = [Decimal(row[column]) for row in rows]
    except InvalidOperation as error:
        raise ValueError(f"CSV column {column!r} must contain exact integers") from error
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


def _scaled_csv_values(
    rows: list[dict[str, str]], columns: str | Sequence[str], factor: float, label: str
) -> FloatArray:
    with np.errstate(over="ignore", invalid="ignore"):
        converted = _csv_values(rows, columns) * factor
    return _finite(converted, f"converted {label}")


def _select_rows(raw: bytes, specification: _SourceSpecification) -> list[dict[str, str]]:
    with io.StringIO(raw.decode("utf-8-sig"), newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader, None)
            _source_header(header, specification)
            assert header is not None
            selected = []
            for values in reader:
                if not values:
                    continue
                if len(values) != len(header):
                    raise ValueError(
                        f"source CSV line {reader.line_num} has {len(values)} cells; "
                        f"expected {len(header)} from the header"
                    )
                row = dict(zip(header, values, strict=True))
                if _matches(row, specification.filters):
                    selected.append(row)
        except csv.Error as error:
            raise ValueError(f"source CSV is malformed near line {reader.line_num}") from error
    if not selected:
        raise ValueError("CSV selection contains no initial rows")
    return selected


def _source_header(header: list[str] | None, specification: _SourceSpecification) -> None:
    if not header or len(set(header)) != len(header) or any(not name for name in header):
        raise ValueError("CSV headers must be present, nonempty and unique")
    columns = specification.columns
    velocity = specification.velocity
    required = {
        columns.particle_id,
        columns.release_time,
        *columns.location,
        *columns.properties.values(),
        *velocity.columns,
        *(velocity.direction_columns or ()),
        *(name for name, _ in specification.filters),
    }
    if missing := required - set(header):
        raise ValueError(f"source CSV is missing mapped or filter columns: {sorted(missing)}")


def _matches(row: dict[str, str], filters: tuple[SourceFilter, ...]) -> bool:
    for name, expected in filters:
        value = row[name]
        if isinstance(expected, str):
            equal = value == expected
        else:
            try:
                actual = Decimal(value)
            except InvalidOperation as error:
                raise ValueError(
                    f"CSV filter column {name!r} must contain numeric values"
                ) from error
            equal = actual.is_finite() and actual == expected
        if not equal:
            return False
    return True


def _configuration(config: Mapping[str, object]) -> _SourceSpecification:
    required = {
        "format_version",
        "table",
        "kind",
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
    optional = {"direction_columns", "filter_equals", "inherit_from"}
    if (
        not isinstance(config, Mapping)
        or set(config) - required - optional
        or required - set(config)
    ):
        raise ValueError("source configuration has missing or unknown keys")
    if config["format_version"] != 2:
        raise ValueError("only source configuration format_version 2 is supported")
    kind = config["kind"]
    if kind not in ("table", "surface"):
        raise ValueError("source kind must be table or surface")
    columns = _source_columns(config["columns"], cast(SourceKind, kind))
    _source_metadata(config["metadata"])
    return _SourceSpecification(
        _text(config["table"], "table"),
        cast(SourceKind, kind),
        columns,
        _source_units(config["units"], config["columns"]),
        _velocity_configuration(config),
        _source_filters(config.get("filter_equals", {})),
        _inheritance_configuration(config.get("inherit_from")),
    )


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"source configuration.{label} must be a nonempty string")
    return value


def _column_sequence(value: object, label: str, count: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f"source configuration.{label} must list {count} CSV columns")
    return tuple(_text(item, label) for item in value)


def _source_columns(raw: object, kind: SourceKind) -> _SourceColumns:
    if not isinstance(raw, Mapping):
        raise ValueError("source configuration.columns must be a mapping")
    location = {"position_m"} if kind == "table" else {"facet_id", "facet_parameter"}
    if set(raw) - {"particle_id", "release_time_s", *location, *_PROPERTIES}:
        raise ValueError("unknown canonical source column")
    if {"particle_id", "release_time_s", *location} - set(raw):
        raise ValueError("CSV must explicitly supply particle ID, release time and location")
    places = (
        _column_sequence(raw["position_m"], "columns.position_m", 2)
        if kind == "table"
        else tuple(_text(raw[name], f"columns.{name}") for name in ("facet_id", "facet_parameter"))
    )
    properties = {name: _text(raw[name], f"columns.{name}") for name in _PROPERTIES if name in raw}
    return _SourceColumns(
        _text(raw["particle_id"], "columns.particle_id"),
        _text(raw["release_time_s"], "columns.release_time_s"),
        places,
        MappingProxyType(properties),
    )


def _source_units(raw: object, raw_columns: object) -> Mapping[str, float]:
    if not isinstance(raw, Mapping):
        raise ValueError("source configuration.units must be a mapping")
    assert isinstance(raw_columns, Mapping)
    if set(raw) - set(raw_columns):
        raise ValueError("units must name supplied columns")
    for name in {"particle_id", "material_id", "facet_id", "facet_parameter"} & set(raw):
        if raw[name] != "1":
            raise ValueError(f"{name} is dimensionless")
    factors = {}
    for name in set(raw_columns) & set(_FIELD_UNITS):
        unit = _text(
            raw.get(name, "1" if name in ("charge_number", "model_weight") else ""), f"units.{name}"
        )
        factors[name] = _unit_factor(unit, _FIELD_UNITS[name], name)
    return MappingProxyType(factors)


def _velocity_configuration(config: Mapping[str, object]) -> _CSVVelocity:
    mode = config["velocity_mode"]
    if mode not in ("vector", "momentum", "speed_direction", "energy_direction"):
        raise ValueError("unsupported source configuration.velocity_mode")
    raw_components = config["components"]
    if not isinstance(raw_components, (list, tuple)) or len(raw_components) not in (2, 3):
        raise ValueError("source configuration.components must list two or three components")
    components = tuple(_text(item, "components") for item in raw_components)
    scalar = mode in ("speed_direction", "energy_direction")
    columns = _column_sequence(
        config["velocity_columns"], "velocity_columns", 1 if scalar else len(components)
    )
    directions = config.get("direction_columns")
    if scalar:
        directions = _column_sequence(directions, "direction_columns", len(components))
    elif directions is not None:
        raise ValueError("vector/momentum input does not use separate direction_columns")
    try:
        basis = np.array(config["basis_to_case"], dtype="<f8", copy=True)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(
            "source configuration.basis_to_case must contain representable numeric values"
        ) from error
    basis.setflags(write=False)
    return _CSVVelocity(
        cast(VelocityMode, mode),
        columns,
        _text(config["velocity_unit"], "velocity_unit"),
        components,
        basis,
        _text(config["frame"], "frame"),
        directions,
    )


def _source_filters(raw: object) -> tuple[SourceFilter, ...]:
    if not isinstance(raw, Mapping):
        raise ValueError("source configuration.filter_equals must be a mapping")
    filters: list[SourceFilter] = []
    for name, expected in raw.items():
        column = _text(name, "filter_equals column")
        if isinstance(expected, bool) or not isinstance(expected, (str, int, float)):
            raise ValueError(f"filter_equals.{column} must be a string or finite number")
        if isinstance(expected, float) and not np.isfinite(expected):
            raise ValueError(f"filter_equals.{column} must be finite")
        number = Decimal(str(expected)) if isinstance(expected, float) else expected
        filters.append((column, number if isinstance(number, str) else Decimal(number)))
    return tuple(filters)


def _inheritance_configuration(inheritance: object) -> str | None:
    if inheritance is None:
        return None
    if (
        not isinstance(inheritance, dict)
        or set(inheritance) != {"table", "join"}
        or inheritance["join"] != "particle_id"
        or not isinstance(inheritance["table"], str)
        or not inheritance["table"]
    ):
        raise ValueError("inherit_from must explicitly name table and join: particle_id")
    return inheritance["table"]


def _source_metadata(metadata: object) -> None:
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
    specification: _SourceSpecification,
    base: RealizedSource | None,
    index: NDArray[np.int64],
) -> NDArray:
    columns = specification.columns.properties
    if name not in columns:
        if base is None:
            raise ValueError(f"CSV requires explicit {name}; inheritance was not requested")
        if np.any(index < 0):
            raise ValueError(f"new particle IDs require explicit {name}; no default is supplied")
        return np.array(getattr(base, name)[index], copy=True)
    if name == "material_id":
        return _integer_column(rows, columns[name], "<i4")
    return _scaled_csv_values(rows, columns[name], specification.unit_factors[name], name)


def read_csv_source(
    path: str | Path, data: DataBundle, config: Mapping[str, object]
) -> RealizedSource:
    """Read explicit realized rows, optionally joining properties by exact IDs.

    A new population can reuse the template geometry/fields when every property
    is supplied. This function returns rows; the canonical writer owns schema
    validation. It never evaluates native expressions or generates random rows.
    """
    return source_from_csv_bytes(Path(path).read_bytes(), data, config)


def source_from_csv_bytes(
    raw: bytes, data: DataBundle, config: Mapping[str, object]
) -> RealizedSource:
    """Resolve one CSV snapshot used by both its caller's hash and the source."""
    specification = _configuration(config)
    return _source_from_rows(_select_rows(raw, specification), data, specification)


def _source_from_rows(
    rows: list[dict[str, str]], data: DataBundle, specification: _SourceSpecification
) -> RealizedSource:
    tables = {source.name: source for source in data.sources}
    inheritance = specification.inherit_table
    base = None if inheritance is None else tables.get(inheritance)
    if inheritance is not None and base is None:
        raise ValueError("inherit_from.table does not name an existing source")
    columns = specification.columns
    ids = _integer_column(rows, columns.particle_id, "<i8")
    if np.unique(ids).size != ids.size:
        raise ValueError("selected initial rows have duplicate particle IDs")
    lookup = {} if base is None else {int(value): i for i, value in enumerate(base.particle_id)}
    index = np.asarray([lookup.get(int(value), -1) for value in ids], dtype="<i8")
    properties = {
        name: _property_values(name, rows, specification, base, index) for name in _PROPERTIES
    }
    release = _scaled_csv_values(
        rows, columns.release_time, specification.unit_factors["release_time_s"], "release_time_s"
    )
    velocity = _csv_velocity(
        rows, properties["mass_kg"], data.coordinate_system, specification.velocity
    )
    common: dict[str, Any] = dict(
        name=specification.table,
        particle_id=ids,
        release_time_s=release,
        velocity_m_s=velocity,
        **properties,
    )
    if specification.kind == "surface":
        return RealizedSurfaceSource(
            facet_id=_integer_column(rows, columns.location[0], "<i8"),
            facet_parameter=_csv_values(rows, columns.location[1]),
            **common,
        )
    position = _scaled_csv_values(
        rows, columns.location, specification.unit_factors["position_m"], "position_m"
    )
    return RealizedTableSource(position_m=position, **common)


def _csv_velocity(
    rows: list[dict[str, str]],
    mass: FloatArray,
    coordinate_system: str,
    specification: _CSVVelocity,
) -> FloatArray:
    mode = specification.mode
    columns = specification.columns
    direction_columns = specification.direction_columns
    direction = None if direction_columns is None else _csv_values(rows, direction_columns)
    return convert_velocity(
        _csv_values(
            rows, columns[0] if mode in ("speed_direction", "energy_direction") else columns
        ),
        mass,
        mode=mode,
        unit=specification.unit,
        components=specification.components,
        coordinate_system=coordinate_system,
        basis_to_case=specification.basis_to_case,
        frame=specification.frame,
        direction=direction,
    )
