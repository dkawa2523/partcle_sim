"""Canonical particle and boundary table conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from particle_tracer_unified.core.catalogs import SUPPORTED_WALL_LAWS

from .legacy import _canonical_choice, _token


def _first_column(
    frame: pd.DataFrame,
    names: Sequence[str],
    *,
    label: str,
    required: bool = False,
    default: Any = None,
) -> pd.Series:
    for name in names:
        if name in frame:
            return frame[name]
    if required:
        raise ValueError(
            f"legacy particle CSV is missing {label}; expected one of {list(names)}"
        )
    return pd.Series(np.full(len(frame), default), index=frame.index)


def _required_particle_columns(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "particle_id": _first_column(
            frame, ("particle_id", "id"), label="particle_id", required=True
        ),
        "release_time_s": _first_column(
            frame,
            ("release_time_s", "release_time", "t0"),
            label="release_time",
            required=True,
        ),
        "mass_kg": _first_column(
            frame, ("mass_kg", "mass"), label="mass", required=True
        ),
        "drag_diameter_m": _first_column(
            frame,
            ("drag_diameter_m", "diameter", "d", "d_eq"),
            label="drag diameter",
            required=True,
        ),
        "charge_C": _first_column(
            frame, ("charge_C", "charge", "q"), label="charge", required=True
        ),
        "source_part_id": _first_column(
            frame,
            ("source_part_id", "part_id_source", "origin_part_id"),
            label="source provenance",
            required=True,
        ),
    }


def _particle_coordinate_aliases(
    spatial_dim: int,
    coordinate_system: str,
) -> dict[str, tuple[str, ...]]:
    coordinate_systems: dict[tuple[int, str], dict[str, tuple[str, ...]]] = {
        (2, "cartesian_xy"): {
            "x_m": ("x_m", "x"),
            "y_m": ("y_m", "y"),
            "vx_mps": ("vx_mps", "vx"),
            "vy_mps": ("vy_mps", "vy"),
        },
        (2, "axisymmetric_rz"): {
            "r_m": ("r_m", "r", "x"),
            "z_m": ("z_m", "z", "y"),
            "vr_mps": ("vr_mps", "vr", "vx"),
            "vz_mps": ("vz_mps", "vz", "vy"),
        },
        (3, "cartesian_xyz"): {
            "x_m": ("x_m", "x"),
            "y_m": ("y_m", "y"),
            "z_m": ("z_m", "z"),
            "vx_mps": ("vx_mps", "vx"),
            "vy_mps": ("vy_mps", "vy"),
            "vz_mps": ("vz_mps", "vz"),
        },
    }
    coordinate_aliases = coordinate_systems.get((spatial_dim, coordinate_system))
    if coordinate_aliases is None:
        raise ValueError(
            f"unsupported coordinate system {coordinate_system!r} for {spatial_dim}D"
        )
    return coordinate_aliases


def _coordinate_particle_columns(
    frame: pd.DataFrame,
    *,
    spatial_dim: int,
    coordinate_system: str,
) -> dict[str, pd.Series]:
    aliases = _particle_coordinate_aliases(spatial_dim, coordinate_system)
    return {
        canonical: _first_column(frame, names, label=canonical, required=True)
        for canonical, names in aliases.items()
    }


def _optional_particle_columns(frame: pd.DataFrame) -> dict[str, pd.Series]:
    aliases = {
        "density_kgm3": ("density_kgm3", "density", "rho_p"),
        "material_id": ("material_id", "particle_material_id"),
        "dep_particle_rel_permittivity": (
            "dep_particle_rel_permittivity",
            "epsr_particle",
        ),
        "thermophoretic_coeff": ("thermophoretic_coeff", "thermo_coeff"),
    }
    return {
        canonical: _first_column(frame, names, label=canonical)
        for canonical, names in aliases.items()
        if any(name in frame for name in names)
    }


def _record_removed_particle_columns(
    frame: pd.DataFrame,
    warnings: list[str],
) -> None:
    removed = [name for name in ("stick_probability", "p_stick") if name in frame]
    if removed:
        warnings.append(
            "dropped legacy particle sticking columns "
            + ", ".join(removed)
            + "; sticking is defined only by boundaries.wall_stick_probability "
            "in schema v2"
        )


def _validate_particle_source_ids(frame: pd.DataFrame) -> None:
    source_ids = np.asarray(
        pd.to_numeric(frame["source_part_id"], errors="coerce"),
        dtype=np.float64,
    )
    invalid = np.isnan(source_ids) | (source_ids <= 0)
    if bool(np.any(invalid)):
        rows = (np.flatnonzero(invalid) + 2).tolist()
        raise ValueError(
            "legacy particle source provenance is missing or invalid; "
            f"source_part_id must be > 0 at CSV rows {rows[:12]}"
        )


def _canonical_particles(
    frame: pd.DataFrame,
    *,
    spatial_dim: int,
    coordinate_system: str,
    warnings: list[str],
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("legacy particle CSV must contain at least one row")
    columns = _required_particle_columns(frame)
    columns.update(
        _coordinate_particle_columns(
            frame,
            spatial_dim=spatial_dim,
            coordinate_system=coordinate_system,
        )
    )
    columns.update(_optional_particle_columns(frame))
    _record_removed_particle_columns(frame, warnings)
    canonical_frame = pd.DataFrame(columns)
    _validate_particle_source_ids(canonical_frame)
    return canonical_frame


def _present(row: Mapping[str, Any] | None, names: Sequence[str]) -> Any:
    if row is None:
        return None
    for name in names:
        if name in row and not pd.isna(row[name]) and str(row[name]).strip() != "":
            return row[name]
    return None


def _resolved_value(
    part: Mapping[str, Any],
    material: Mapping[str, Any] | None,
    names: Sequence[str],
    default: Any,
) -> Any:
    value = _present(part, names)
    if value is None:
        value = _present(material, names)
    return default if value is None else value


def _resolved_text(
    part: Mapping[str, Any],
    material: Mapping[str, Any] | None,
    names: Sequence[str],
    default: str,
    *,
    label: str,
) -> str:
    for row in (part, material):
        if row is None:
            continue
        for name in names:
            if name not in row or pd.isna(row[name]):
                continue
            value = str(row[name]).strip()
            if not value:
                raise ValueError(f"legacy {label} must not be blank")
            return value
    value = str(default).strip()
    if not value:
        raise ValueError(f"legacy {label} is required and must not be blank")
    return value


_WALL_LAW_ALIASES = {
    "bounce": "specular",
    "open": "escape",
    "outflow": "escape",
    "passthrough": "pass_through",
    "diffuse": "cosine_diffuse",
    "diffuse_scattering": "cosine_diffuse",
    "mixed_diffuse_specular": "mixed_specular_diffuse",
    "critical_sticking": "critical_sticking_velocity",
    "field_support_exit": "escape",
}

_INTERNAL_BOUNDARY_ROLE = "internal"

_BOUNDARY_ROLE_ALIASES = {
    "boundary": "wall",
    "input": "inlet",
    "open": "outlet",
    "outflow": "outlet",
    "interior": _INTERNAL_BOUNDARY_ROLE,
    "passthrough": _INTERNAL_BOUNDARY_ROLE,
    "pass_through": _INTERNAL_BOUNDARY_ROLE,
    "field_support_boundary": "field_support",
}


def _boundary_materials(
    materials: pd.DataFrame | None,
) -> dict[int, Mapping[str, Any]]:
    lookup: dict[int, Mapping[str, Any]] = {}
    if materials is not None:
        for _, row in materials.iterrows():
            lookup[int(row.get("material_id", 0) or 0)] = row.to_dict()
    return lookup


def _boundary_law_and_default_role(
    part: Mapping[str, Any],
    material: Mapping[str, Any] | None,
    wall_config: Mapping[str, Any],
    *,
    part_id: int,
) -> tuple[str, str]:
    raw_law = _resolved_text(
        part,
        material,
        ("wall_law",),
        str(wall_config.get("default_mode", wall_config.get("mode", ""))),
        label=f"wall_law for part_id={part_id}",
    )
    original_law = _token(raw_law)
    law = _canonical_choice(
        raw_law,
        canonical=tuple(SUPPORTED_WALL_LAWS),
        aliases=_WALL_LAW_ALIASES,
        label=f"wall_law for part_id={part_id}",
    )
    default_role = {
        "escape": "outlet",
        "pass_through": _INTERNAL_BOUNDARY_ROLE,
    }.get(law, "wall")
    if original_law == "field_support_exit":
        default_role = "field_support"
    return law, default_role


def _boundary_float(
    part: Mapping[str, Any],
    material: Mapping[str, Any] | None,
    names: Sequence[str],
    default: Any,
) -> float:
    return float(_resolved_value(part, material, names, default))


def _canonical_boundary_row(
    raw_row: pd.Series,
    row_index: Any,
    materials: Mapping[int, Mapping[str, Any]],
    wall_config: Mapping[str, Any],
) -> dict[str, Any]:
    part = raw_row.to_dict()
    part_id = int(part.get("part_id", 0) or 0)
    if part_id <= 0:
        raise ValueError(
            f"legacy part wall CSV has invalid part_id at row {int(row_index) + 2}"
        )
    material_id = int(part.get("material_id", 0) or 0)
    material = materials.get(material_id)
    law, default_role = _boundary_law_and_default_role(
        part,
        material,
        wall_config,
        part_id=part_id,
    )
    role = _canonical_choice(
        _resolved_text(
            part,
            material,
            ("role",),
            default_role,
            label=f"role for part_id={part_id}",
        ),
        canonical=("wall", "inlet", "outlet", "internal", "field_support"),
        aliases=_BOUNDARY_ROLE_ALIASES,
        label=f"role for part_id={part_id}",
    )
    part_name = _resolved_text(
        part,
        None,
        ("part_name",),
        f"part_{part_id}",
        label=f"part_name for part_id={part_id}",
    )
    material_name = _resolved_text(
        part,
        material,
        ("material_name",),
        f"material_{material_id}",
        label=f"material_name for part_id={part_id}",
    )
    return {
        "part_id": part_id,
        "part_name": part_name,
        "role": role,
        "material_id": material_id,
        "material_name": material_name,
        "wall_law": law,
        "wall_stick_probability": _boundary_float(
            part,
            material,
            ("wall_stick_probability", "stick_probability"),
            wall_config.get(
                "stick_probability",
                wall_config.get("default_stick_probability", 0.0),
            ),
        ),
        "wall_restitution": _boundary_float(
            part,
            material,
            ("wall_restitution", "restitution"),
            wall_config.get("restitution", 1.0),
        ),
        "wall_diffuse_fraction": _boundary_float(
            part,
            material,
            ("wall_diffuse_fraction", "diffuse_fraction"),
            wall_config.get("diffuse_fraction", 0.0),
        ),
        "wall_critical_sticking_velocity_mps": _boundary_float(
            part,
            material,
            (
                "wall_critical_sticking_velocity_mps",
                "critical_sticking_velocity_mps",
            ),
            wall_config.get("critical_sticking_velocity_mps", 0.0),
        ),
    }


def _validate_unique_boundary_ids(frame: pd.DataFrame) -> None:
    if frame["part_id"].duplicated().any():
        duplicate_ids = sorted(
            frame.loc[frame["part_id"].duplicated(False), "part_id"].unique().tolist()
        )
        raise ValueError(
            f"legacy part wall CSV contains duplicate part IDs: {duplicate_ids}"
        )


def _canonical_boundaries(
    walls: pd.DataFrame,
    materials: pd.DataFrame | None,
    wall_config: Mapping[str, Any],
) -> pd.DataFrame:
    if walls.empty:
        raise ValueError("legacy part wall CSV must contain at least one row")
    materials_by_id = _boundary_materials(materials)
    rows = [
        _canonical_boundary_row(raw_row, row_index, materials_by_id, wall_config)
        for row_index, raw_row in walls.iterrows()
    ]
    frame = pd.DataFrame(rows)
    _validate_unique_boundary_ids(frame)
    return frame.sort_values("part_id", kind="stable").reset_index(drop=True)
