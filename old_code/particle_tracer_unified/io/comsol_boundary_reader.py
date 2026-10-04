from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from particle_tracer_unified.core.catalogs import SUPPORTED_WALL_LAWS
from particle_tracer_unified.core.datamodel import PartWallRow, PartWallTable

ALLOWED_BOUNDARY_ROLES = {
    "wall",
    "inlet",
    "outlet",
    "field_support",
    "internal",
}

CANONICAL_BOUNDARY_COLUMNS = frozenset(
    {
        "part_id",
        "part_name",
        "comsol_entity_id",
        "role",
        "wall_law",
        "wall_stick_probability",
        "wall_restitution",
        "wall_diffuse_fraction",
        "wall_critical_sticking_velocity_mps",
        "material_id",
        "material_name",
    }
)
CANONICAL_BOUNDARY_OPTIONAL_COLUMNS = frozenset({"selection_name", "metadata_json"})


@dataclass(frozen=True)
class ComsolBoundaryMapRow:
    solver_part_id: int
    comsol_geom_entity_id: int
    selection_name: str
    boundary_type: str
    material: str


@dataclass(frozen=True)
class ComsolWallLawRow:
    solver_part_id: int
    wall_type: str
    stick_probability: float
    restitution_n: float
    mixed_diffuse_fraction: float
    material_id: int
    critical_sticking_velocity_mps: float
    material_name: str
    material_metadata: Mapping[str, Any] = field(default_factory=dict)


def _first(row: Mapping[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def _exact_text(
    row: Mapping[str, str],
    name: str,
    *,
    path: Path,
    line_no: int,
    default: str | None = None,
) -> str:
    """Read an identifier without silently trimming canonical CSV data."""

    value = row.get(name)
    if value is None or value == "":
        if default is not None:
            return default
        raise ValueError(f"{name} is required in {path}:{line_no}")
    text = str(value)
    if text != text.strip():
        raise ValueError(
            f"{name} must not contain leading or trailing whitespace "
            f"in {path}:{line_no}"
        )
    return text


def _float(row: Mapping[str, str], *names: str, default: float = np.nan) -> float:
    value = _first(row, *names, default="")
    return float(default) if value == "" else float(value)


def _int(row: Mapping[str, str], *names: str) -> int:
    value = _first(row, *names, default="")
    if value == "":
        raise ValueError(f"{names[0]} is required")
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ValueError(f"{names[0]} must contain an integer, got {value!r}") from exc
    if not np.isfinite(numeric) or numeric != np.floor(numeric):
        raise ValueError(f"{names[0]} must contain an integer, got {value!r}")
    return int(numeric)


def _read_canonical_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        columns = set(header)
        duplicates = sorted({name for name in header if header.count(name) > 1})
        if duplicates:
            raise ValueError(f"{path} contains duplicate columns: {duplicates}")
        missing = sorted(CANONICAL_BOUNDARY_COLUMNS - columns)
        if missing:
            raise ValueError(f"{path} is missing canonical boundary columns: {missing}")
        unknown = sorted(
            columns - CANONICAL_BOUNDARY_COLUMNS - CANONICAL_BOUNDARY_OPTIONAL_COLUMNS
        )
        if unknown:
            raise ValueError(
                f"{path} contains unknown canonical boundary columns: {unknown}"
            )
        return [dict(row) for row in reader]


def _material_metadata(
    row: Mapping[str, str], *, path: Path, line_no: int
) -> Mapping[str, Any]:
    raw = _first(row, "metadata_json", default="")
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid metadata_json in {path}:{line_no}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"metadata_json must contain an object in {path}:{line_no}")
    return dict(payload)


def comsol_entity_to_solver_part_id(
    rows: list[ComsolBoundaryMapRow],
    *,
    path: Path | None = None,
) -> dict[int, int]:
    """Return the one authoritative COMSOL-entity to solver-part bijection."""

    entity_to_part: dict[int, int] = {}
    seen_parts: set[int] = set()
    location = f" in {path}" if path is not None else ""
    for row in rows:
        part_id = int(row.solver_part_id)
        entity_id = int(row.comsol_geom_entity_id)
        if part_id in seen_parts:
            raise ValueError(f"Duplicate solver_part_id={part_id}{location}")
        if entity_id in entity_to_part:
            raise ValueError(f"Duplicate comsol_entity_id={entity_id}{location}")
        seen_parts.add(part_id)
        entity_to_part[entity_id] = part_id
    return entity_to_part


def remap_comsol_boundary_entity_ids(
    entity_ids: np.ndarray,
    entity_to_part_id: Mapping[int, int],
    *,
    context: str,
) -> np.ndarray:
    """Map an edge or triangle entity array into the solver part-ID space."""

    values = np.asarray(entity_ids)
    if values.dtype.kind not in "iu":
        raise ValueError(f"{context} must contain integer COMSOL entity IDs")
    integer_values = values.astype(np.int64, copy=False)
    if np.any(integer_values <= 0):
        raise ValueError(f"{context} must contain positive COMSOL entity IDs")
    present = {int(value) for value in np.unique(integer_values)}
    missing = sorted(present - set(entity_to_part_id))
    if missing:
        raise ValueError(f"{context} is missing comsol_entity_id values: {missing}")

    result = np.empty(integer_values.shape, dtype=np.int32)
    max_part_id = int(np.iinfo(np.int32).max)
    for entity_id in present:
        part_id = int(entity_to_part_id[entity_id])
        if not 0 < part_id <= max_part_id:
            raise ValueError(
                f"solver_part_id={part_id} for comsol_entity_id={entity_id} "
                "is outside the positive int32 range"
            )
        result[integer_values == entity_id] = part_id
    return result


def _wall_coefficients(
    row: Mapping[str, str],
    *,
    path: Path,
    line_no: int,
) -> tuple[float, float, float, float]:
    stick = _float(row, "wall_stick_probability")
    restitution = _float(row, "wall_restitution")
    diffuse = _float(row, "wall_diffuse_fraction")
    critical = _float(row, "wall_critical_sticking_velocity_mps")
    coefficients = {
        "stick_probability": stick,
        "restitution": restitution,
        "diffuse_fraction": diffuse,
        "critical_sticking_velocity_mps": critical,
    }
    for name, value in coefficients.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite in {path}:{line_no}")
    if not 0.0 <= stick <= 1.0:
        raise ValueError(f"stick_probability must be in [0, 1] in {path}:{line_no}")
    if restitution < 0.0:
        raise ValueError(f"restitution must be non-negative in {path}:{line_no}")
    if not 0.0 <= diffuse <= 1.0:
        raise ValueError(f"diffuse_fraction must be in [0, 1] in {path}:{line_no}")
    if critical < 0.0:
        raise ValueError(
            f"critical_sticking_velocity_mps must be non-negative in {path}:{line_no}"
        )
    return stick, restitution, diffuse, critical


def _parse_boundary_row(
    row: Mapping[str, str],
    *,
    path: Path,
    line_no: int,
) -> tuple[ComsolBoundaryMapRow, ComsolWallLawRow]:
    part_id = _int(row, "part_id")
    entity_id = _int(row, "comsol_entity_id")
    role = _exact_text(row, "role", path=path, line_no=line_no)
    if part_id <= 0:
        raise ValueError(f"part_id must be positive in {path}:{line_no}")
    if entity_id <= 0:
        raise ValueError(f"comsol_entity_id must be positive in {path}:{line_no}")
    if role not in ALLOWED_BOUNDARY_ROLES:
        raise ValueError(
            f"role must be one of {sorted(ALLOWED_BOUNDARY_ROLES)} "
            f"in {path}:{line_no}, got {role!r}"
        )
    part_name = _exact_text(row, "part_name", path=path, line_no=line_no)
    law = _exact_text(row, "wall_law", path=path, line_no=line_no)
    if law not in SUPPORTED_WALL_LAWS:
        expected = ", ".join(sorted(SUPPORTED_WALL_LAWS))
        raise ValueError(
            f"Unsupported wall_law in {path}:{line_no} {law!r}; "
            f"expected one of {expected}"
        )

    stick, restitution, diffuse, critical = _wall_coefficients(
        row,
        path=path,
        line_no=line_no,
    )
    material_id = _int(row, "material_id")
    if material_id < 0:
        raise ValueError(f"material_id must be non-negative in {path}:{line_no}")
    material_name = _exact_text(row, "material_name", path=path, line_no=line_no)
    selection_name = _exact_text(
        row,
        "selection_name",
        path=path,
        line_no=line_no,
        default=part_name,
    )
    material_metadata = _material_metadata(row, path=path, line_no=line_no)
    return (
        ComsolBoundaryMapRow(
            solver_part_id=part_id,
            comsol_geom_entity_id=entity_id,
            selection_name=selection_name,
            boundary_type=role,
            material=material_name,
        ),
        ComsolWallLawRow(
            solver_part_id=part_id,
            wall_type=law,
            stick_probability=stick,
            restitution_n=restitution,
            mixed_diffuse_fraction=diffuse,
            material_id=material_id,
            critical_sticking_velocity_mps=critical,
            material_name=material_name,
            material_metadata=material_metadata,
        ),
    )


def read_comsol_boundaries(
    path: str | Path,
) -> tuple[list[ComsolBoundaryMapRow], list[ComsolWallLawRow]]:
    """Read the single-file COMSOL boundary and wall-law contract."""

    boundary_path = Path(path)
    source_rows = _read_canonical_rows(boundary_path)
    boundary_rows: list[ComsolBoundaryMapRow] = []
    wall_rows: list[ComsolWallLawRow] = []
    for line_no, row in enumerate(source_rows, start=2):
        boundary, wall = _parse_boundary_row(
            row,
            path=boundary_path,
            line_no=line_no,
        )
        boundary_rows.append(boundary)
        wall_rows.append(wall)
    if not boundary_rows:
        raise ValueError(f"{boundary_path} must contain at least one boundary row")
    comsol_entity_to_solver_part_id(boundary_rows, path=boundary_path)
    return boundary_rows, wall_rows


def _diffuse_fraction(row: ComsolWallLawRow) -> float:
    if row.wall_type == "cosine_diffuse":
        return 1.0
    if row.wall_type != "mixed_specular_diffuse":
        return 0.0
    if np.isfinite(row.mixed_diffuse_fraction):
        return float(row.mixed_diffuse_fraction)
    return 0.5


def _part_wall_row(
    row: ComsolWallLawRow,
    boundary: ComsolBoundaryMapRow | None,
) -> PartWallRow:
    part_id = int(row.solver_part_id)
    default_name = f"comsol_boundary_{part_id}"
    if boundary is None:
        part_name, role = default_name, "wall"
    else:
        part_name = boundary.selection_name or default_name
        role = boundary.boundary_type
    stick_probability = (
        1.0 if row.wall_type == "stick" else float(row.stick_probability)
    )
    return PartWallRow(
        part_id=part_id,
        part_name=str(part_name),
        role=str(role),
        material_id=int(row.material_id),
        material_name=str(row.material_name),
        wall_law=str(row.wall_type),
        wall_restitution=float(row.restitution_n),
        wall_diffuse_fraction=_diffuse_fraction(row),
        wall_stick_probability=float(np.clip(stick_probability, 0.0, 1.0)),
        wall_critical_sticking_velocity_mps=float(row.critical_sticking_velocity_mps),
        metadata=dict(row.material_metadata),
    )


def wall_laws_to_boundaries(
    rows: list[ComsolWallLawRow],
    boundary_rows: list[ComsolBoundaryMapRow] | None = None,
) -> PartWallTable:
    boundary_by_part = {int(row.solver_part_id): row for row in (boundary_rows or ())}
    part_rows = [
        _part_wall_row(row, boundary_by_part.get(int(row.solver_part_id)))
        for row in rows
    ]
    metadata: dict[str, object] = {
        "source": "comsol_wall_laws",
        "material_metadata_by_part": {
            str(int(row.solver_part_id)): dict(row.material_metadata)
            for row in rows
            if row.material_metadata
        },
    }
    if boundary_rows is not None:
        metadata["comsol_boundary_map"] = [asdict(row) for row in boundary_rows]
    return PartWallTable(
        rows=tuple(sorted(part_rows, key=lambda item: item.part_id)), metadata=metadata
    )


def validate_wall_law_coverage(
    boundary_rows: list[ComsolBoundaryMapRow],
    wall_law_rows: list[ComsolWallLawRow],
    *,
    exact: bool = False,
) -> None:
    boundary_parts = {int(row.solver_part_id) for row in boundary_rows}
    wall_parts = {int(row.solver_part_id) for row in wall_law_rows}
    missing = sorted(boundary_parts - wall_parts)
    if missing:
        raise ValueError(
            f"COMSOL faithful wall law is missing solver_part_id values: {missing}"
        )
    extra = sorted(wall_parts - boundary_parts)
    if exact and extra:
        raise ValueError(
            f"COMSOL faithful wall law has unmapped solver_part_id values: {extra}"
        )


def validate_geometry_boundary_coverage(
    geometry: Any,
    boundary_rows: list[ComsolBoundaryMapRow],
    wall_law_rows: list[ComsolWallLawRow],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Compare explicit geometry parts with the complete boundary/wall map."""

    geom = getattr(geometry, "geometry", geometry)
    arrays = (
        getattr(geom, "boundary_edge_part_ids", None),
        getattr(geom, "boundary_triangle_part_ids", None),
    )
    geometry_parts: set[int] = set()
    for raw in arrays:
        if raw is not None:
            geometry_parts.update(
                int(value) for value in np.unique(np.asarray(raw)) if int(value) > 0
            )
    mapped_parts = {int(row.solver_part_id) for row in boundary_rows}
    wall_parts = {int(row.solver_part_id) for row in wall_law_rows}
    report = {
        "passed": bool(geometry_parts == mapped_parts == wall_parts),
        "geometry_part_ids": sorted(geometry_parts),
        "mapped_part_ids": sorted(mapped_parts),
        "wall_part_ids": sorted(wall_parts),
        "missing_boundary_rows": sorted(geometry_parts - mapped_parts),
        "stale_boundary_rows": sorted(mapped_parts - geometry_parts),
        "missing_wall_laws": sorted(geometry_parts - wall_parts),
        "stale_wall_laws": sorted(wall_parts - geometry_parts),
    }
    if strict and not report["passed"]:
        raise ValueError(
            "COMSOL geometry/boundary/wall part coverage must match exactly: "
            f"missing_boundary_rows={report['missing_boundary_rows']}, "
            f"stale_boundary_rows={report['stale_boundary_rows']}, "
            f"missing_wall_laws={report['missing_wall_laws']}, "
            f"stale_wall_laws={report['stale_wall_laws']}"
        )
    return report


__all__ = (
    "ALLOWED_BOUNDARY_ROLES",
    "CANONICAL_BOUNDARY_COLUMNS",
    "CANONICAL_BOUNDARY_OPTIONAL_COLUMNS",
    "ComsolBoundaryMapRow",
    "ComsolWallLawRow",
    "comsol_entity_to_solver_part_id",
    "read_comsol_boundaries",
    "remap_comsol_boundary_entity_ids",
    "validate_geometry_boundary_coverage",
    "validate_wall_law_coverage",
    "wall_laws_to_boundaries",
)
