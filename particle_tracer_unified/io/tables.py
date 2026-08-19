"""Convert validated canonical CSV tables into domain arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from particle_tracer_unified.core.datamodel import (
    ParticleTable,
    PartWallRow,
    PartWallTable,
)

from .canonical_tables import (
    _coordinate_columns,
    validate_boundaries_csv,
    validate_particles_csv,
)


def _optional_float(frame: pd.DataFrame, name: str, default: float) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), float(default), dtype=np.float64)
    return frame[name].to_numpy(dtype=np.float64)


def _optional_int(frame: pd.DataFrame, name: str, default: int) -> np.ndarray:
    if name not in frame:
        return np.full(len(frame), int(default), dtype=np.int64)
    return frame[name].to_numpy(dtype=np.int64)


def load_particles_csv(
    path: Path, spatial_dim: int, coordinate_system: str
) -> ParticleTable:
    source = Path(path).resolve()
    frame = validate_particles_csv(
        source,
        spatial_dim=int(spatial_dim),
        coordinate_system=str(coordinate_system),
    )
    position_columns, velocity_columns = _coordinate_columns(
        spatial_dim, coordinate_system
    )
    count = len(frame)
    return ParticleTable(
        spatial_dim=int(spatial_dim),
        particle_id=frame["particle_id"].to_numpy(dtype=np.int64),
        position=frame[list(position_columns)].to_numpy(dtype=np.float64),
        velocity=frame[list(velocity_columns)].to_numpy(dtype=np.float64),
        release_time=frame["release_time_s"].to_numpy(dtype=np.float64),
        mass=frame["mass_kg"].to_numpy(dtype=np.float64),
        diameter=frame["drag_diameter_m"].to_numpy(dtype=np.float64),
        density=_optional_float(frame, "density_kgm3", np.nan),
        charge=frame["charge_C"].to_numpy(dtype=np.float64),
        source_part_id=frame["source_part_id"].to_numpy(dtype=np.int64),
        material_id=_optional_int(frame, "material_id", 0),
        dep_particle_rel_permittivity=_optional_float(
            frame, "dep_particle_rel_permittivity", np.nan
        ),
        thermophoretic_coeff=_optional_float(frame, "thermophoretic_coeff", np.nan),
        metadata={"path": str(source), "schema_version": 2, "row_count": int(count)},
    )


def _metadata(value: object) -> dict[str, object]:
    if pd.isna(value) or str(value).strip() == "":
        return {}
    parsed = json.loads(str(value))
    return dict(parsed)


def load_boundaries_csv(path: Path) -> PartWallTable:
    source = Path(path).resolve()
    frame = validate_boundaries_csv(source)
    rows = tuple(
        PartWallRow(
            part_id=int(row.part_id),
            part_name=str(row.part_name),
            role=str(row.role),
            material_id=int(row.material_id),
            material_name=str(row.material_name),
            wall_law=str(row.wall_law),
            wall_stick_probability=float(row.wall_stick_probability),
            wall_restitution=float(row.wall_restitution),
            wall_diffuse_fraction=float(row.wall_diffuse_fraction),
            wall_critical_sticking_velocity_mps=float(
                np.asarray(row.wall_critical_sticking_velocity_mps).item()
            ),
            metadata=_metadata(getattr(row, "metadata_json", "")),
        )
        for row in frame.itertuples(index=False)
    )
    return PartWallTable(
        rows=rows,
        metadata={"path": str(source), "schema_version": 2, "row_count": len(rows)},
    )


__all__ = ("load_boundaries_csv", "load_particles_csv")
