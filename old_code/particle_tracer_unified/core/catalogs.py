from __future__ import annotations

import numpy as np

from .datamodel import (
    PartWallRow,
    PartWallTable,
    WallCatalog,
    WallPartModel,
)

SUPPORTED_WALL_LAWS = frozenset(
    {
        "stick",
        "freeze",
        "absorb",
        "escape",
        "pass_through",
        "specular",
        "cosine_diffuse",
        "mixed_specular_diffuse",
        "critical_sticking_velocity",
    }
)


def normalize_wall_law_name(value: object, *, context: str = "wall law") -> str:
    key = str(value).strip().lower()
    if not key or key in {"nan", "none", "null"}:
        raise ValueError(f"{context} must be a supported wall law")
    if key not in SUPPORTED_WALL_LAWS:
        expected = ", ".join(sorted(SUPPORTED_WALL_LAWS))
        raise ValueError(f"Unsupported {context} {value!r}; expected one of {expected}")
    return key


def is_internal_pass_through(model: WallPartModel) -> bool:
    """Whether a wall model represents a transparent interior interface."""

    return bool(
        str(model.law_name) == "pass_through"
        and str(model.metadata.get("role", "")) == "internal"
    )


def _validated_wall_coefficients(
    row: PartWallRow,
    part_id: int,
) -> tuple[float, float, float, float]:
    stick_probability = float(row.wall_stick_probability)
    restitution = float(row.wall_restitution)
    diffuse_fraction = float(row.wall_diffuse_fraction)
    critical_velocity = float(row.wall_critical_sticking_velocity_mps)
    values = (stick_probability, restitution, diffuse_fraction, critical_velocity)
    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"Boundary part_id={part_id} has non-finite wall coefficients")
    if not 0.0 <= stick_probability <= 1.0:
        raise ValueError(
            f"Boundary part_id={part_id} stick probability must be in [0, 1]"
        )
    if not 0.0 <= diffuse_fraction <= 1.0:
        raise ValueError(
            f"Boundary part_id={part_id} diffuse fraction must be in [0, 1]"
        )
    if restitution < 0.0 or critical_velocity < 0.0:
        raise ValueError(
            f"Boundary part_id={part_id} wall coefficients must be non-negative"
        )
    return values


def _wall_part_model(row: PartWallRow, part_id: int) -> WallPartModel:
    stick, restitution, diffuse, critical_velocity = _validated_wall_coefficients(
        row,
        part_id,
    )
    normalized_law = normalize_wall_law_name(
        row.wall_law,
        context=f"wall law for part_id={part_id}",
    )
    return WallPartModel(
        part_id=part_id,
        part_name=str(row.part_name),
        material_id=int(row.material_id),
        material_name=str(row.material_name),
        law_name=normalized_law,
        stick_probability=stick,
        restitution=restitution,
        diffuse_fraction=diffuse,
        critical_sticking_velocity_mps=critical_velocity,
        metadata={"role": str(row.role), **dict(row.metadata)},
    )


def build_wall_catalog(
    walls: PartWallTable | None,
) -> WallCatalog:
    """Build the boundary catalog without inherited defaults or fallbacks."""

    if walls is None or not walls.rows:
        raise ValueError("A non-empty canonical boundaries.csv is required")
    part_models: list[WallPartModel] = []
    seen: set[int] = set()
    for row in walls.rows:
        part_id = int(row.part_id)
        if part_id in seen:
            raise ValueError(f"Duplicate boundary part_id={part_id}")
        seen.add(part_id)
        part_models.append(_wall_part_model(row, part_id))
    return WallCatalog(
        part_models=tuple(part_models),
        metadata={"wall_part_count": len(part_models)},
    )


def resolve_step_wall_model(
    wall_catalog: WallCatalog | None,
    part_id: int,
) -> WallPartModel:
    if wall_catalog is None:
        raise ValueError("Boundary catalog is required before applying a wall law")
    return wall_catalog.model_for_part(int(part_id))


__all__ = (
    "SUPPORTED_WALL_LAWS",
    "build_wall_catalog",
    "is_internal_pass_through",
    "normalize_wall_law_name",
    "resolve_step_wall_model",
)
