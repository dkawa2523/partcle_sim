"""Validate and canonicalize explicit inputs to the COMSOL case builder."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd


def required_positive_float(value: Any, *, context: str) -> float:
    if value is None or isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{context} is required and must be positive and finite")
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context} is required and must be positive and finite"
        ) from exc
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{context} is required and must be positive and finite")
    return resolved


@dataclass(frozen=True, slots=True)
class GeometryOnlyBuild:
    kind: Literal["geometry_only"] = "geometry_only"


FIELD_STORAGE_REGULAR_GRID = "regular_grid"
FIELD_STORAGE_MESH_NATIVE = "mesh_native"


@dataclass(frozen=True, slots=True)
class RunnableBuild:
    """One runnable COMSOL build with exactly one declared field source.

    ``field_bundle_path`` resamples the solution onto a regular grid.
    ``field_node_samples_path`` keeps it on the COMSOL mesh, evaluated at the
    mesh vertices.  Exactly one is set; the storage kind follows from that
    choice and is never inferred from file contents.
    """

    release_table_path: Path
    boundaries_path: Path
    model_name: str
    study: str
    dataset: str
    solution: str
    solution_number: int
    drag_law: str
    solver_dt_s: float
    solver_t_end_s: float
    field_bundle_path: Path | None = None
    field_node_samples_path: Path | None = None
    kind: Literal["runnable"] = "runnable"

    @property
    def field_storage(self) -> str:
        if self.field_node_samples_path is not None:
            return FIELD_STORAGE_MESH_NATIVE
        return FIELD_STORAGE_REGULAR_GRID


def _positive_solution_number(value: int | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("solution_number must be a positive integer")
    return value


def validate_runnable_inputs(
    *,
    geometry_only: bool,
    field_bundle_path: Path | None,
    release_table_path: Path | None,
    field_node_samples_path: Path | None = None,
    boundaries_path: Path | None,
    model_name: str | None,
    study: str | None,
    dataset: str | None,
    solution: str | None,
    solution_number: int | None,
    drag_law: str | None,
    solver_dt_s: float | None,
    solver_t_end_s: float | None,
) -> GeometryOnlyBuild | RunnableBuild:
    """Validate the inputs that distinguish geometry-only from runnable output."""

    if geometry_only:
        if field_bundle_path is not None or field_node_samples_path is not None:
            raise ValueError(
                "geometry_only cannot be combined with a field bundle or node samples"
            )
        return GeometryOnlyBuild()
    if field_bundle_path is not None and field_node_samples_path is not None:
        raise ValueError(
            "declare exactly one field source: --field-bundle for a resampled "
            "regular grid, or --field-node-samples for the COMSOL mesh itself"
        )
    if field_bundle_path is None and field_node_samples_path is None:
        raise ValueError(
            "COMSOL case generation requires --field-bundle or "
            "--field-node-samples; use --geometry-only to build geometry only"
        )
    missing = [
        name
        for name, value in (
            ("release_table_path", release_table_path),
            ("boundaries_path", boundaries_path),
            ("model_name", model_name),
            ("study", study),
            ("dataset", dataset),
            ("solution", solution),
            ("solution_number", solution_number),
            ("drag_law", drag_law),
        )
        if value is None or str(value).strip() == ""
    ]
    if missing:
        raise ValueError(f"runnable COMSOL case requires explicit inputs: {missing}")
    return RunnableBuild(
        field_bundle_path=(
            None if field_bundle_path is None else Path(field_bundle_path)
        ),
        field_node_samples_path=(
            None if field_node_samples_path is None else Path(field_node_samples_path)
        ),
        release_table_path=cast(Path, release_table_path),
        boundaries_path=cast(Path, boundaries_path),
        model_name=str(model_name),
        study=str(study),
        dataset=str(dataset),
        solution=str(solution),
        solution_number=_positive_solution_number(solution_number),
        drag_law=str(drag_law),
        solver_dt_s=required_positive_float(solver_dt_s, context="solver_dt_s"),
        solver_t_end_s=required_positive_float(
            solver_t_end_s, context="solver_t_end_s"
        ),
    )


def copy_explicit_input(source: Path, destination: Path) -> None:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source != destination:
        shutil.copy2(source, destination)


def load_json_mapping(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def canonical_boundary_table(
    source: Path, *, geometry_entity_ids: list[int]
) -> pd.DataFrame:
    from particle_tracer_unified.io.comsol_boundary_reader import read_comsol_boundaries

    boundary_rows, _wall_rows = read_comsol_boundaries(source)
    declared = {int(row.comsol_geom_entity_id) for row in boundary_rows}
    expected = set(map(int, geometry_entity_ids))
    if declared != expected:
        raise ValueError(
            "boundaries.csv must explicitly cover every generated geometry part: "
            f"missing={sorted(expected - declared)}, "
            f"stale={sorted(declared - expected)}"
        )
    return pd.read_csv(source)


def canonical_release_table(source: Path, *, coordinate_system: str) -> pd.DataFrame:
    from particle_tracer_unified.io.canonical_tables import validate_particles_csv

    return validate_particles_csv(
        source, spatial_dim=2, coordinate_system=coordinate_system
    )


__all__ = (
    "GeometryOnlyBuild",
    "RunnableBuild",
    "canonical_boundary_table",
    "canonical_release_table",
    "copy_explicit_input",
    "load_json_mapping",
    "required_positive_float",
    "validate_runnable_inputs",
)
