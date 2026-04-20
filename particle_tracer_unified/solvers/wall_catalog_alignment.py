from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

from ..core.datamodel import WallCatalog, WallPartModel


def _read_csv_by_key(path: Path, key: str) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        out: dict[int, dict[str, str]] = {}
        for row in rows:
            raw = row.get(key, "")
            try:
                out[int(float(str(raw).strip()))] = {str(k): str(v) for k, v in row.items()}
            except (TypeError, ValueError):
                continue
        return out


def _read_material_inventory(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _float_close(left: object, right: object, *, tol: float = 1.0e-12) -> bool:
    try:
        return abs(float(left) - float(right)) <= float(tol)
    except (TypeError, ValueError):
        return False


def _current_model_row(model: WallPartModel, *, explicit: bool) -> dict[str, object]:
    return {
        "current_explicit_wall_catalog": int(bool(explicit)),
        "current_part_name": str(model.part_name),
        "current_material_id": int(model.material_id),
        "current_material_name": str(model.material_name),
        "current_wall_law": str(model.law_name),
        "current_wall_restitution": float(model.restitution),
        "current_wall_diffuse_fraction": float(model.diffuse_fraction),
        "current_wall_stick_probability": float(model.stick_probability),
    }


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _adjacent_domain_count(value: object) -> int:
    text = str(value).strip()
    if not text:
        return 0
    return len([part for part in text.split(";") if part.strip()])


def _classify_entity_role(
    *,
    part_id: int,
    active: bool,
    boundary: Mapping[str, str],
    review: Mapping[str, str],
    model: WallPartModel,
) -> tuple[str, str]:
    names = " ".join(
        str(value)
        for value in (
            boundary.get("solver_part_name", ""),
            boundary.get("solver_material_name", ""),
            review.get("part_name", ""),
            review.get("material_name", ""),
            model.part_name,
            model.material_name,
        )
    ).lower()
    if int(part_id) >= 9000 or "field_support" in names:
        return "support_only", "synthetic field-support or open-boundary entity"
    if "axis_symmetry" in names:
        return ("active_axis" if active else "axis"), "axis-symmetry boundary"
    if active:
        return "active_wall", "solver collision boundary"
    adjacent_count = _adjacent_domain_count(
        boundary.get("adjacent_domain_ids", review.get("adjacent_domain_ids", ""))
    )
    if adjacent_count >= 2:
        return "internal_interface", "COMSOL interface between two domains; not a solver collision wall"
    return "inactive", "COMSOL entity outside the active solver boundary set"


def _review_mismatch_fields(review: Mapping[str, str], model: WallPartModel) -> list[str]:
    fields: list[str] = []
    comparisons = (
        ("material_id", int(model.material_id), "int"),
        ("material_name", str(model.material_name), "str"),
        ("wall_law", str(model.law_name), "str"),
        ("wall_restitution", float(model.restitution), "float"),
        ("wall_diffuse_fraction", float(model.diffuse_fraction), "float"),
        ("wall_stick_probability", float(model.stick_probability), "float"),
    )
    for key, current, kind in comparisons:
        if key not in review:
            continue
        old = str(review.get(key, "")).strip()
        if kind == "int":
            try:
                same = int(float(old)) == int(current)
            except ValueError:
                same = False
        elif kind == "float":
            same = _float_close(old, current)
        else:
            same = old == str(current)
        if not same:
            fields.append(key)
    return fields


def build_wall_catalog_alignment(
    *,
    generated_dir: Path | None,
    wall_catalog: WallCatalog | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if generated_dir is None:
        return {"enabled": 1, "generated_dir_found": 0, "review_flags": ["generated_dir_not_found"]}, []
    generated = Path(generated_dir)
    boundary_path = generated / "comsol_boundary_entity_mapping.csv"
    review_path = generated / "wall_catalog_review.csv"
    material_path = generated / "material_inventory.json"
    boundary_rows = _read_csv_by_key(boundary_path, "solver_part_id")
    review_rows = _read_csv_by_key(review_path, "part_id")
    material_inventory = _read_material_inventory(material_path)

    explicit_models = wall_catalog.as_lookup() if wall_catalog is not None else {}
    default_model = wall_catalog.default_model if wall_catalog is not None else None
    part_ids = set(boundary_rows) | set(review_rows) | set(explicit_models)
    rows: list[dict[str, object]] = []
    mismatch_count = 0
    default_count = 0
    active_default_count = 0
    inactive_default_count = 0
    unknown_comsol_material_count = 0
    active_unknown_comsol_material_count = 0
    inactive_unknown_comsol_material_count = 0
    active_count = 0
    comsol_known_count = 0
    role_counts: dict[str, int] = {}
    for part_id in sorted(part_ids):
        explicit = int(part_id) in explicit_models
        model = explicit_models.get(int(part_id), default_model)
        if model is None:
            continue
        boundary = boundary_rows.get(int(part_id), {})
        review = review_rows.get(int(part_id), {})
        mismatch_fields = _review_mismatch_fields(review, model) if review else []
        if mismatch_fields:
            mismatch_count += 1
        if not explicit:
            default_count += 1
        active = _truthy(boundary.get("active_in_solver_boundary", review.get("active_in_solver_boundary", "")))
        if active:
            active_count += 1
        role, role_reason = _classify_entity_role(
            part_id=int(part_id),
            active=active,
            boundary=boundary,
            review=review,
            model=model,
        )
        role_counts[role] = int(role_counts.get(role, 0) + 1)
        comsol_material = str(
            boundary.get("comsol_material_name", review.get("comsol_material_name", ""))
        ).strip()
        if comsol_material and comsol_material != "not_exported_from_mphtxt":
            comsol_known_count += 1
        else:
            unknown_comsol_material_count += 1
            if active:
                active_unknown_comsol_material_count += 1
            else:
                inactive_unknown_comsol_material_count += 1
        status = "matched"
        if not review and not boundary:
            status = "wall_catalog_only"
        elif not explicit and active:
            status = "active_uses_default_wall_catalog"
        elif not explicit:
            status = "classified_inactive_default"
        elif mismatch_fields:
            status = "review_mismatch"
        if not explicit:
            if active:
                active_default_count += 1
            else:
                inactive_default_count += 1
        row = {
            "part_id": int(part_id),
            "solver_entity_role": role,
            "solver_entity_role_reason": role_reason,
            "alignment_status": status,
            "mismatch_fields": "|".join(mismatch_fields),
            "comsol_edge_entity_id": boundary.get("comsol_edge_entity_id", review.get("comsol_edge_entity_id", "")),
            "active_in_solver_boundary": boundary.get("active_in_solver_boundary", review.get("active_in_solver_boundary", "")),
            "adjacent_domain_ids": boundary.get("adjacent_domain_ids", review.get("adjacent_domain_ids", "")),
            "comsol_material_name": comsol_material,
            "mapping_solver_part_name": boundary.get("solver_part_name", ""),
            "mapping_solver_material_name": boundary.get("solver_material_name", ""),
            "review_part_name": review.get("part_name", ""),
            "review_material_id": review.get("material_id", ""),
            "review_material_name": review.get("material_name", ""),
            "review_wall_law": review.get("wall_law", ""),
            "review_wall_restitution": review.get("wall_restitution", ""),
            "review_wall_diffuse_fraction": review.get("wall_diffuse_fraction", ""),
            "review_wall_stick_probability": review.get("wall_stick_probability", ""),
            "x_min_m": boundary.get("x_min_m", review.get("x_min_m", "")),
            "x_max_m": boundary.get("x_max_m", review.get("x_max_m", "")),
            "y_min_m": boundary.get("y_min_m", review.get("y_min_m", "")),
            "y_max_m": boundary.get("y_max_m", review.get("y_max_m", "")),
            **_current_model_row(model, explicit=explicit),
        }
        rows.append(row)

    material_count = 0
    materials = material_inventory.get("materials", []) if isinstance(material_inventory, Mapping) else []
    if isinstance(materials, list):
        material_count = int(len(materials))
    flags: list[str] = []
    info_flags: list[str] = []
    if mismatch_count:
        flags.append("wall_catalog_review_mismatch")
    if active_default_count:
        flags.append("active_solver_boundary_uses_default_wall_catalog")
    if active_unknown_comsol_material_count:
        flags.append("active_solver_boundary_without_comsol_material_name")
    if inactive_default_count:
        info_flags.append("inactive_entities_use_default_wall_catalog")
    if inactive_unknown_comsol_material_count:
        info_flags.append("inactive_entities_without_comsol_material_name")
    summary = {
        "enabled": 1,
        "generated_dir_found": int(generated.exists()),
        "generated_dir": str(generated.resolve()) if generated.exists() else str(generated),
        "boundary_mapping_file_found": int(boundary_path.exists()),
        "wall_catalog_review_file_found": int(review_path.exists()),
        "material_inventory_file_found": int(material_path.exists()),
        "material_inventory_count": int(material_count),
        "boundary_mapping_part_count": int(len(boundary_rows)),
        "wall_catalog_review_part_count": int(len(review_rows)),
        "current_explicit_wall_catalog_part_count": int(len(explicit_models)),
        "alignment_row_count": int(len(rows)),
        "active_solver_boundary_count": int(active_count),
        "solver_entity_role_counts": {str(key): int(value) for key, value in sorted(role_counts.items())},
        "comsol_material_known_part_count": int(comsol_known_count),
        "comsol_material_unknown_part_count": int(unknown_comsol_material_count),
        "default_wall_catalog_part_count": int(default_count),
        "active_default_wall_catalog_part_count": int(active_default_count),
        "inactive_default_wall_catalog_part_count": int(inactive_default_count),
        "active_comsol_material_unknown_part_count": int(active_unknown_comsol_material_count),
        "inactive_comsol_material_unknown_part_count": int(inactive_unknown_comsol_material_count),
        "review_mismatch_count": int(mismatch_count),
        "review_flags": flags,
        "info_flags": info_flags,
    }
    return summary, rows


WALL_CATALOG_ALIGNMENT_COLUMNS = (
    "part_id",
    "solver_entity_role",
    "solver_entity_role_reason",
    "alignment_status",
    "mismatch_fields",
    "comsol_edge_entity_id",
    "active_in_solver_boundary",
    "adjacent_domain_ids",
    "comsol_material_name",
    "mapping_solver_part_name",
    "mapping_solver_material_name",
    "review_part_name",
    "review_material_id",
    "review_material_name",
    "review_wall_law",
    "review_wall_restitution",
    "review_wall_diffuse_fraction",
    "review_wall_stick_probability",
    "current_explicit_wall_catalog",
    "current_part_name",
    "current_material_id",
    "current_material_name",
    "current_wall_law",
    "current_wall_restitution",
    "current_wall_diffuse_fraction",
    "current_wall_stick_probability",
    "x_min_m",
    "x_max_m",
    "y_min_m",
    "y_max_m",
)


def write_wall_catalog_alignment_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(WALL_CATALOG_ALIGNMENT_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "WALL_CATALOG_ALIGNMENT_COLUMNS",
    "build_wall_catalog_alignment",
    "write_wall_catalog_alignment_csv",
]
