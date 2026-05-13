from __future__ import annotations

import csv
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import numpy as np

from ..core.datamodel import MaterialRow, MaterialTable, PartWallRow, PartWallTable


ALLOWED_WALL_TYPES = {
    "bounce",
    "specular",
    "stick",
    "freeze",
    "disappear",
    "absorb",
    "pass_through",
    "passthrough",
    "diffuse",
    "mixed_specular_diffuse",
}


@dataclass(frozen=True)
class ComsolBoundaryMapRow:
    solver_part_id: int
    comsol_geom_entity_id: int
    selection_name: str
    boundary_type: str
    wall_node: str
    material: str
    notes: str = ""


@dataclass(frozen=True)
class ComsolWallLawRow:
    solver_part_id: int
    wall_type: str
    stick_probability: float
    restitution_n: float
    restitution_t: float
    diffuse_temperature: float
    mixed_diffuse_fraction: float
    material_id: str
    notes: str = ""


def _first(row: Mapping[str, str], *names: str, default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def _float(row: Mapping[str, str], *names: str, default: float = np.nan) -> float:
    value = _first(row, *names, default="")
    return float(default) if value == "" else float(value)


def _int(row: Mapping[str, str], *names: str, default: int = 0) -> int:
    value = _first(row, *names, default="")
    return int(default) if value == "" else int(float(value))


def _mixed_diffuse_fraction(row: Mapping[str, str]) -> float:
    explicit = _float(row, "diffuse_probability", "diffuse_fraction", "wall_diffuse_fraction", default=np.nan)
    if np.isfinite(explicit):
        return float(np.clip(explicit, 0.0, 1.0))
    specular = _float(row, "specular_probability", "specular_fraction", "reflection_probability", "gamma", default=np.nan)
    if np.isfinite(specular):
        return float(1.0 - np.clip(specular, 0.0, 1.0))
    return float("nan")


def _read_dicts(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _check_duplicate_part_ids(rows: list[object], *, attr: str, path: Path) -> None:
    seen: set[int] = set()
    for row in rows:
        part_id = int(getattr(row, attr))
        if part_id in seen:
            raise ValueError(f"Duplicate solver_part_id={part_id} in {path}")
        seen.add(part_id)


def read_comsol_boundary_map(path: str | Path, *, strict: bool = True) -> list[ComsolBoundaryMapRow]:
    boundary_path = Path(path)
    rows = []
    for row in _read_dicts(boundary_path):
        solver_part_id = _int(row, "solver_part_id", "part_id")
        comsol_entity = _int(
            row,
            "comsol_geom_entity_id",
            "comsol_entity_id",
            "comsol_boundary_id",
            "comsol_edge_entity_id",
        )
        if strict and solver_part_id <= 0:
            raise ValueError(f"solver_part_id must be positive in {boundary_path}")
        rows.append(
            ComsolBoundaryMapRow(
                solver_part_id=solver_part_id,
                comsol_geom_entity_id=comsol_entity,
                selection_name=_first(row, "selection_name", "selection", default=""),
                boundary_type=_first(row, "boundary_type", default="wall"),
                wall_node=_first(row, "wall_node", default=""),
                material=_first(row, "material", "material_name", default=""),
                notes=_first(row, "notes", default=""),
            )
        )
    _check_duplicate_part_ids(rows, attr="solver_part_id", path=boundary_path)
    return rows


def read_comsol_wall_laws(path: str | Path, *, strict: bool = True) -> list[ComsolWallLawRow]:
    wall_path = Path(path)
    rows = []
    for row in _read_dicts(wall_path):
        wall_type = _first(row, "wall_type", "wall_law", default="").lower()
        if wall_type == "freeze":
            wall_type = "stick"
        if strict and wall_type not in ALLOWED_WALL_TYPES:
            raise ValueError(f"Unknown wall_type={wall_type!r} in {wall_path}")
        parsed = ComsolWallLawRow(
            solver_part_id=_int(row, "solver_part_id", "part_id"),
            wall_type=wall_type,
            stick_probability=_float(row, "stick_probability", "wall_stick_probability", default=0.0),
            restitution_n=_float(row, "restitution_n", "wall_restitution", default=1.0),
            restitution_t=_float(row, "restitution_t", default=1.0),
            diffuse_temperature=_float(row, "diffuse_temperature", default=np.nan),
            mixed_diffuse_fraction=_mixed_diffuse_fraction(row),
            material_id=_first(row, "material_id", "material", "material_name", default="0"),
            notes=_first(row, "notes", default=""),
        )
        reflecting_wall = parsed.wall_type in {"bounce", "specular", "diffuse", "mixed_specular_diffuse"}
        if strict and reflecting_wall and np.isfinite(parsed.restitution_t) and not np.isclose(float(parsed.restitution_t), 1.0):
            raise ValueError(f"COMSOL tangential restitution is not supported in {wall_path}")
        if strict and reflecting_wall and np.isfinite(parsed.diffuse_temperature):
            raise ValueError(f"COMSOL diffuse_temperature/thermal reemission is not supported in {wall_path}")
        rows.append(parsed)
    _check_duplicate_part_ids(rows, attr="solver_part_id", path=wall_path)
    return rows


def wall_laws_to_tables(
    rows: list[ComsolWallLawRow],
    boundary_rows: list[ComsolBoundaryMapRow] | None = None,
) -> tuple[MaterialTable, PartWallTable]:
    boundary_by_part = {int(row.solver_part_id): row for row in (boundary_rows or [])}
    material_name_to_id: dict[str, int] = {"": 0, "0": 0, "none": 0}
    material_rows: dict[int, MaterialRow] = {}
    part_rows = []

    def material_id_for(raw: str) -> tuple[int, str]:
        text = str(raw).strip()
        if text == "":
            return 0, ""
        try:
            return int(text), text
        except ValueError:
            if text not in material_name_to_id:
                material_name_to_id[text] = max(material_name_to_id.values(), default=0) + 1
            return int(material_name_to_id[text]), text

    for row in rows:
        boundary = boundary_by_part.get(int(row.solver_part_id))
        material_id, material_name = material_id_for(row.material_id)
        if boundary is not None and boundary.material and not material_name:
            material_name = boundary.material
        part_name = f"comsol_boundary_{int(row.solver_part_id)}"
        if boundary is not None and boundary.selection_name:
            part_name = boundary.selection_name
        wall_law = row.wall_type
        restitution = float(row.restitution_n)
        mixed_diffuse = float(row.mixed_diffuse_fraction) if np.isfinite(row.mixed_diffuse_fraction) else 0.5
        diffuse_fraction = 1.0 if wall_law == "diffuse" else (mixed_diffuse if wall_law == "mixed_specular_diffuse" else 0.0)
        stick_probability = 1.0 if wall_law == "stick" else float(row.stick_probability)
        material_rows.setdefault(
            int(material_id),
            MaterialRow(
                material_id=int(material_id),
                material_name=str(material_name),
                source_law="explicit_csv",
                source_speed_scale=1.0,
                wall_law=str(wall_law),
                wall_restitution=float(restitution),
                wall_diffuse_fraction=float(diffuse_fraction),
                wall_stick_probability=float(np.clip(stick_probability, 0.0, 1.0)),
            ),
        )
        part_rows.append(
            PartWallRow(
                part_id=int(row.solver_part_id),
                part_name=str(part_name),
                material_id=int(material_id),
                material_name=str(material_name),
                wall_law=str(wall_law),
                wall_restitution=float(restitution),
                wall_diffuse_fraction=float(diffuse_fraction),
                wall_stick_probability=float(np.clip(stick_probability, 0.0, 1.0)),
            )
        )
    metadata: dict[str, object] = {"source": "comsol_wall_laws"}
    if boundary_rows is not None:
        metadata["comsol_boundary_map"] = [asdict(row) for row in boundary_rows]
    return (
        MaterialTable(rows=tuple(material_rows[key] for key in sorted(material_rows)), metadata=metadata),
        PartWallTable(rows=tuple(sorted(part_rows, key=lambda item: item.part_id)), metadata=metadata),
    )


def validate_wall_law_coverage(
    boundary_rows: list[ComsolBoundaryMapRow],
    wall_law_rows: list[ComsolWallLawRow],
) -> None:
    boundary_parts = {int(row.solver_part_id) for row in boundary_rows if str(row.boundary_type).lower() in {"wall", "boundary"}}
    wall_parts = {int(row.solver_part_id) for row in wall_law_rows}
    missing = sorted(boundary_parts - wall_parts)
    if missing:
        raise ValueError(f"COMSOL faithful wall law is missing solver_part_id values: {missing}")


__all__ = (
    "ALLOWED_WALL_TYPES",
    "ComsolBoundaryMapRow",
    "ComsolWallLawRow",
    "read_comsol_boundary_map",
    "read_comsol_wall_laws",
    "validate_wall_law_coverage",
    "wall_laws_to_tables",
)
