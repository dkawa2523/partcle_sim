"""Validation for the two canonical v0.2 CSV inputs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from particle_tracer_unified.core.catalogs import SUPPORTED_WALL_LAWS

_PARTICLE_BASE_COLUMNS = {
    "particle_id",
    "release_time_s",
    "mass_kg",
    "drag_diameter_m",
    "charge_C",
    "source_part_id",
}
_PARTICLE_OPTIONAL_COLUMNS = {
    "density_kgm3",
    "material_id",
    "dep_particle_rel_permittivity",
    "thermophoretic_coeff",
    "metadata_json",
}
_BOUNDARY_COLUMNS = {
    "part_id",
    "part_name",
    "role",
    "material_id",
    "material_name",
    "wall_law",
    "wall_stick_probability",
    "wall_restitution",
    "wall_diffuse_fraction",
    "wall_critical_sticking_velocity_mps",
}
_BOUNDARY_OPTIONAL_COLUMNS = {"metadata_json"}
_BOUNDARY_ROLES = {"wall", "inlet", "outlet", "internal", "field_support"}


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"{label} CSV does not exist: {source}")
    try:
        frame = pd.read_csv(source)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise ValueError(f"could not read {label} CSV {source}: {exc}") from exc
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"{label} CSV has duplicate columns: {duplicates}")
    if frame.empty:
        raise ValueError(f"{label} CSV must contain at least one row: {source}")
    return frame


def _coordinate_columns(
    spatial_dim: int,
    coordinate_system: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    dim = int(spatial_dim)
    coordinate = str(coordinate_system)
    if dim == 2 and coordinate == "cartesian_xy":
        return ("x_m", "y_m"), ("vx_mps", "vy_mps")
    if dim == 2 and coordinate == "axisymmetric_rz":
        return ("r_m", "z_m"), ("vr_mps", "vz_mps")
    if dim == 3 and coordinate == "cartesian_xyz":
        return ("x_m", "y_m", "z_m"), ("vx_mps", "vy_mps", "vz_mps")
    raise ValueError(
        "unsupported coordinate contract: "
        f"spatial_dim={dim}, coordinate_system={coordinate!r}"
    )


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    required_set = set(required)
    missing = sorted(required_set.difference(map(str, frame.columns)))
    if missing:
        raise ValueError(
            f"{label} CSV is missing required columns: {', '.join(missing)}"
        )


def _reject_columns(frame: pd.DataFrame, allowed: Iterable[str], label: str) -> None:
    allowed_set = set(allowed)
    unknown = sorted(
        str(name) for name in frame.columns if str(name) not in allowed_set
    )
    if unknown:
        raise ValueError(f"{label} CSV has unknown columns: {', '.join(unknown)}")


def _numeric(frame: pd.DataFrame, name: str, label: str) -> np.ndarray:
    try:
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.{name} must be numeric") from exc
    if not np.all(np.isfinite(values)):
        rows = (np.flatnonzero(~np.isfinite(values)) + 2).tolist()
        raise ValueError(
            f"{label}.{name} has non-finite values at CSV rows {rows[:12]}"
        )
    return values


def _integer_values(frame: pd.DataFrame, name: str, label: str) -> np.ndarray:
    values = _numeric(frame, name, label)
    if not np.all(values == np.floor(values)):
        raise ValueError(f"{label}.{name} must contain integers")
    return values.astype(np.int64)


def _reject_numeric_values(
    invalid: np.ndarray,
    message: str,
    *,
    report_rows: bool = False,
) -> None:
    if not np.any(invalid):
        return
    if report_rows:
        rows = (np.flatnonzero(invalid) + 2).tolist()
        message = f"{message}; invalid CSV rows {rows[:12]}"
    raise ValueError(message)


def _validate_numeric_bound(
    frame: pd.DataFrame,
    name: str,
    *,
    minimum: float,
    inclusive: bool,
    message: str,
) -> None:
    values = _numeric(frame, name, "particles")
    invalid = values < minimum if inclusive else values <= minimum
    _reject_numeric_values(invalid, message)


def _validate_metadata_json(frame: pd.DataFrame, label: str) -> None:
    if "metadata_json" not in frame:
        return
    for row_index, value in enumerate(frame["metadata_json"].tolist(), start=2):
        if pd.isna(value) or str(value).strip() == "":
            continue
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{label}.metadata_json is invalid JSON at CSV row {row_index}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"{label}.metadata_json must be a JSON object at CSV row {row_index}"
            )


def _exact_text_values(frame: pd.DataFrame, name: str, label: str) -> tuple[str, ...]:
    """Return non-empty text without repairing whitespace or letter case."""

    values: list[str] = []
    empty_rows: list[int] = []
    padded_rows: list[int] = []
    for row_index, value in enumerate(frame[name].tolist(), start=2):
        if pd.isna(value):
            empty_rows.append(row_index)
            continue
        text = str(value)
        if text == "":
            empty_rows.append(row_index)
            continue
        if text != text.strip():
            padded_rows.append(row_index)
        values.append(text)
    if empty_rows:
        raise ValueError(
            f"{label}.{name} must not be empty; invalid CSV rows {empty_rows[:12]}"
        )
    if padded_rows:
        raise ValueError(
            f"{label}.{name} must not contain leading or trailing whitespace; "
            f"invalid CSV rows {padded_rows[:12]}"
        )
    return tuple(values)


def validate_particles_csv(
    path: str | Path, *, spatial_dim: int, coordinate_system: str
) -> pd.DataFrame:
    """Validate and return a canonical particle table.

    Column suffixes encode SI units; aliases and implicit physical defaults are
    intentionally not accepted here.  The returned frame is a copy owned by the
    caller and is not written back to disk.
    """

    frame = _read_csv(Path(path), "particles")
    position_columns, velocity_columns = _coordinate_columns(
        spatial_dim, coordinate_system
    )
    required = _PARTICLE_BASE_COLUMNS.union(position_columns).union(velocity_columns)
    allowed = required.union(_PARTICLE_OPTIONAL_COLUMNS)
    _require_columns(frame, required, "particles")
    _reject_columns(frame, allowed, "particles")

    particle_ids = _integer_values(frame, "particle_id", "particles")
    _reject_numeric_values(particle_ids < 0, "particles.particle_id must be >= 0")
    if np.unique(particle_ids).size != particle_ids.size:
        raise ValueError("particles.particle_id values must be unique")
    source_ids = _integer_values(frame, "source_part_id", "particles")
    _reject_numeric_values(
        source_ids <= 0,
        "particles.source_part_id must be > 0",
        report_rows=True,
    )

    motion_values = {
        name: _numeric(frame, name, "particles")
        for name in (*position_columns, *velocity_columns, "charge_C")
    }
    if coordinate_system == "axisymmetric_rz":
        _reject_numeric_values(
            motion_values["r_m"] < 0.0,
            "particles.r_m must be >= 0 for axisymmetric_rz",
            report_rows=True,
        )
    _validate_numeric_bound(
        frame,
        "release_time_s",
        minimum=0.0,
        inclusive=True,
        message="particles.release_time_s must be >= 0",
    )
    for name in ("mass_kg", "drag_diameter_m"):
        _validate_numeric_bound(
            frame,
            name,
            minimum=0.0,
            inclusive=False,
            message=f"particles.{name} must be > 0",
        )
    if "density_kgm3" in frame:
        _validate_numeric_bound(
            frame,
            "density_kgm3",
            minimum=0.0,
            inclusive=False,
            message="particles.density_kgm3 must be > 0",
        )
    if "material_id" in frame:
        material_ids = _integer_values(frame, "material_id", "particles")
        _reject_numeric_values(material_ids < 0, "particles.material_id must be >= 0")
    for name in ("dep_particle_rel_permittivity", "thermophoretic_coeff"):
        if name in frame:
            _numeric(frame, name, "particles")
    _validate_metadata_json(frame, "particles")
    return frame


def _validate_boundary_identity(frame: pd.DataFrame) -> tuple[str, ...]:
    part_ids = _integer_values(frame, "part_id", "boundaries")
    _reject_numeric_values(part_ids <= 0, "boundaries.part_id must be > 0")
    if np.unique(part_ids).size != part_ids.size:
        raise ValueError("boundaries.part_id values must be unique")
    material_ids = _integer_values(frame, "material_id", "boundaries")
    _reject_numeric_values(material_ids < 0, "boundaries.material_id must be >= 0")

    text_columns = {
        name: _exact_text_values(frame, name, "boundaries")
        for name in ("part_name", "material_name", "role", "wall_law")
    }
    roles = text_columns["role"]
    bad_roles = sorted(set(roles).difference(_BOUNDARY_ROLES))
    if bad_roles:
        raise ValueError(
            f"boundaries.role has unsupported values: {', '.join(bad_roles)}"
        )
    return text_columns["wall_law"]


def _validate_boundary_wall_contract(
    frame: pd.DataFrame, laws: tuple[str, ...]
) -> None:
    bad_laws = sorted(set(laws).difference(SUPPORTED_WALL_LAWS))
    if bad_laws:
        raise ValueError(
            f"boundaries.wall_law has unsupported values: {', '.join(bad_laws)}"
        )
    probability = _numeric(frame, "wall_stick_probability", "boundaries")
    diffuse = _numeric(frame, "wall_diffuse_fraction", "boundaries")
    _reject_numeric_values(
        (probability < 0.0) | (probability > 1.0),
        "boundaries.wall_stick_probability must be in [0, 1]",
    )
    _reject_numeric_values(
        (diffuse < 0.0) | (diffuse > 1.0),
        "boundaries.wall_diffuse_fraction must be in [0, 1]",
    )
    for name in ("wall_restitution", "wall_critical_sticking_velocity_mps"):
        values = _numeric(frame, name, "boundaries")
        _reject_numeric_values(values < 0.0, f"boundaries.{name} must be >= 0")


def validate_boundaries_csv(path: str | Path) -> pd.DataFrame:
    """Validate and return the unified part/material/wall table."""

    frame = _read_csv(Path(path), "boundaries")
    _require_columns(frame, _BOUNDARY_COLUMNS, "boundaries")
    _reject_columns(
        frame, _BOUNDARY_COLUMNS.union(_BOUNDARY_OPTIONAL_COLUMNS), "boundaries"
    )
    laws = _validate_boundary_identity(frame)
    _validate_boundary_wall_contract(frame, laws)
    _validate_metadata_json(frame, "boundaries")
    return frame


__all__ = ["validate_boundaries_csv", "validate_particles_csv"]
