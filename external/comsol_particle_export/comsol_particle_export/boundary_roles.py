from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


ROLE_PRIORITY = {
    "pair_continuity": 30,
    "outlet_freeze": 20,
    "wall_bounce": 10,
}


def _read_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {str(col).strip().lower(): str(col) for col in frame.columns}
    for name in names:
        found = lower.get(str(name).strip().lower())
        if found is not None:
            return found
    return None


def _load_boundary_map(path: str | Path | None) -> dict[int, int]:
    if path is None:
        return {}
    frame = pd.read_csv(path)
    left = _first_column(
        frame,
        (
            "comsol_api_selection_entity_id",
            "comsol_boundary_id",
            "comsol_entity_id",
            "comsol_edge_entity_id",
            "boundary_id",
        ),
    )
    right = _first_column(frame, ("solver_part_id", "part_id"))
    if left is None or right is None:
        raise ValueError("boundary map must contain COMSOL entity and solver_part_id columns")
    out: dict[int, int] = {}
    for c, s in zip(pd.to_numeric(frame[left], errors="coerce"), pd.to_numeric(frame[right], errors="coerce")):
        if pd.notna(c) and pd.notna(s):
            out[int(c)] = int(s)
    return out


def _as_entities(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _is_particle_physics(feature: Mapping[str, Any]) -> bool:
    tag = str(feature.get("physics_tag", "")).strip().lower()
    label = str(feature.get("physics_label", "")).strip().lower()
    ptype = str(feature.get("physics_type", "")).strip().lower()
    return tag.startswith(("fpt", "pt")) or "particle tracing" in label or "particletracing" in ptype.replace(" ", "")


def _wall_condition(feature: Mapping[str, Any]) -> str:
    values = feature.get("property_values", {})
    if not isinstance(values, Mapping):
        values = feature.get("known_settings", {})
    if not isinstance(values, Mapping):
        return ""
    return str(values.get("WallCondition", values.get("wallcondition", ""))).strip().lower()


def _restitution(feature: Mapping[str, Any]) -> float:
    values = feature.get("property_values", {})
    if not isinstance(values, Mapping):
        return 1.0
    try:
        return float(str(values.get("e", "1")).replace("[", " ").split()[0])
    except (TypeError, ValueError):
        return 1.0


def _role_for_feature(feature: Mapping[str, Any]) -> tuple[str, str, bool, float] | None:
    ftype = str(feature.get("type", "")).strip().lower()
    if not _is_particle_physics(feature):
        return None
    if ftype == "paircontinuity":
        return "pair_continuity", "pass_through", False, 1.0
    if ftype == "outlet" and _wall_condition(feature) == "freeze":
        return "outlet_freeze", "stick", True, 0.0
    if ftype == "wall" and _wall_condition(feature) == "bounce":
        return "wall_bounce", "specular", True, _restitution(feature)
    return None


def _choose_role(existing: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    if existing is None:
        return candidate
    old_priority = ROLE_PRIORITY.get(str(existing.get("expected_role", "")), 0)
    new_priority = ROLE_PRIORITY.get(str(candidate.get("expected_role", "")), 0)
    if new_priority >= old_priority:
        merged_flags = sorted(set(str(existing.get("role_flags", "")).split("|")) | set(str(candidate.get("role_flags", "")).split("|")))
        candidate = dict(candidate)
        candidate["role_flags"] = "|".join(flag for flag in merged_flags if flag)
        return candidate
    existing = dict(existing)
    flags = sorted(set(str(existing.get("role_flags", "")).split("|")) | {str(candidate.get("expected_role", ""))})
    existing["role_flags"] = "|".join(flag for flag in flags if flag)
    return existing


def derive_boundary_roles(
    *,
    raw_export_dir: str | Path,
    boundary_map_csv: str | Path | None = None,
    part_walls_csv: str | Path | None = None,
    out_dir: str | Path | None = None,
    write_part_walls_csv: str | Path | None = None,
    write_materials_csv: str | Path | None = None,
) -> dict[str, Any]:
    raw_dir = Path(raw_export_dir)
    inventory = _read_json(raw_dir / "physics_feature_inventory.json")
    boundary_map = _load_boundary_map(boundary_map_csv)
    features = inventory.get("features", [])
    if not isinstance(features, list):
        features = []

    expected: dict[int, dict[str, Any]] = {}
    for feature in features:
        if not isinstance(feature, Mapping):
            continue
        role = _role_for_feature(feature)
        if role is None:
            continue
        expected_role, wall_law, active_collision, restitution = role
        for entity in _as_entities(feature.get("selection_entities", [])):
            part_id = int(boundary_map.get(int(entity), int(entity)))
            row = {
                "solver_part_id": part_id,
                "comsol_entity_id": int(entity),
                "expected_role": expected_role,
                "role_flags": expected_role,
                "expected_wall_law": wall_law,
                "expected_active_collision": int(bool(active_collision)),
                "expected_restitution": float(restitution),
                "feature_tag": str(feature.get("feature_tag", "")),
                "feature_label": str(feature.get("label", "")),
                "feature_type": str(feature.get("type", "")),
            }
            expected[part_id] = _choose_role(expected.get(part_id), row)

    current = pd.DataFrame()
    if part_walls_csv is not None and Path(part_walls_csv).exists():
        current = pd.read_csv(part_walls_csv)
    current_by_part = {}
    if not current.empty and "part_id" in current.columns:
        for _, row in current.iterrows():
            current_by_part[int(row.get("part_id", 0))] = row.to_dict()

    rows = []
    for part_id in sorted(set(expected) | set(current_by_part)):
        exp = expected.get(part_id, {})
        cur = current_by_part.get(part_id, {})
        expected_law = str(exp.get("expected_wall_law", ""))
        current_law = str(cur.get("wall_law", ""))
        rows.append(
            {
                "solver_part_id": int(part_id),
                "comsol_entity_id": exp.get("comsol_entity_id", ""),
                "expected_role": exp.get("expected_role", "unclassified"),
                "role_flags": exp.get("role_flags", ""),
                "expected_wall_law": expected_law,
                "current_wall_law": current_law,
                "expected_active_collision": exp.get("expected_active_collision", ""),
                "current_part_wall_present": int(bool(cur)),
                "role_match": int((not expected_law) or expected_law == current_law),
                "feature_tag": exp.get("feature_tag", ""),
                "feature_label": exp.get("feature_label", ""),
                "feature_type": exp.get("feature_type", ""),
            }
        )
    alignment = pd.DataFrame(rows)

    if write_part_walls_csv is not None:
        wall_rows = []
        for part_id, exp in sorted(expected.items()):
            role = str(exp.get("expected_role", ""))
            law = str(exp.get("expected_wall_law", "specular"))
            if role == "pair_continuity":
                material_id = 3
                material_name = "comsol_pair_continuity"
                part_name = f"comsol_pair_continuity_{part_id}"
            elif role == "outlet_freeze":
                material_id = 2
                material_name = "comsol_freeze"
                part_name = f"comsol_outlet_freeze_{part_id}"
            else:
                material_id = 1
                material_name = "comsol_bounce"
                part_name = f"comsol_boundary_{part_id}"
            wall_rows.append(
                {
                    "part_id": int(part_id),
                    "part_name": part_name,
                    "material_id": material_id,
                    "material_name": material_name,
                    "wall_law": law,
                    "wall_restitution": float(exp.get("expected_restitution", 1.0)),
                    "wall_diffuse_fraction": 0.0,
                    "wall_stick_probability": 1.0 if law == "stick" else 0.0,
                }
            )
        out_path = Path(write_part_walls_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(wall_rows).to_csv(out_path, index=False)

    if write_materials_csv is not None:
        materials = pd.DataFrame(
            [
                {
                    "material_id": 1,
                    "material_name": "comsol_bounce",
                    "source_law": "explicit_csv",
                    "source_speed_scale": 1.0,
                    "wall_law": "specular",
                    "wall_restitution": 1.0,
                    "wall_diffuse_fraction": 0.0,
                    "wall_stick_probability": 0.0,
                },
                {
                    "material_id": 2,
                    "material_name": "comsol_freeze",
                    "source_law": "explicit_csv",
                    "source_speed_scale": 1.0,
                    "wall_law": "stick",
                    "wall_restitution": 0.0,
                    "wall_diffuse_fraction": 0.0,
                    "wall_stick_probability": 1.0,
                },
                {
                    "material_id": 3,
                    "material_name": "comsol_pair_continuity",
                    "source_law": "explicit_csv",
                    "source_speed_scale": 1.0,
                    "wall_law": "pass_through",
                    "wall_restitution": 1.0,
                    "wall_diffuse_fraction": 0.0,
                    "wall_stick_probability": 0.0,
                },
            ]
        )
        materials_path = Path(write_materials_csv)
        materials_path.parent.mkdir(parents=True, exist_ok=True)
        materials.to_csv(materials_path, index=False)

    mismatch_count = int((alignment["role_match"] == 0).sum()) if not alignment.empty else 0
    summary = {
        "available": True,
        "raw_export_dir": str(raw_dir),
        "boundary_map_csv": str(boundary_map_csv) if boundary_map_csv is not None else "",
        "part_walls_csv": str(part_walls_csv) if part_walls_csv is not None else "",
        "expected_part_count": int(len(expected)),
        "alignment_row_count": int(len(alignment)),
        "mismatch_count": mismatch_count,
        "expected_role_counts": {
            str(k): int(v)
            for k, v in alignment["expected_role"].value_counts(dropna=False).sort_index().items()
        }
        if not alignment.empty
        else {},
        "expected_wall_law_counts": {
            str(k): int(v)
            for k, v in alignment["expected_wall_law"].value_counts(dropna=False).sort_index().items()
        }
        if not alignment.empty
        else {},
    }

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        alignment.to_csv(out / "boundary_role_alignment.csv", index=False)
        (out / "boundary_role_alignment.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summary["outputs"] = {
            "boundary_role_alignment_csv": str(out / "boundary_role_alignment.csv"),
            "boundary_role_alignment_json": str(out / "boundary_role_alignment.json"),
        }
    return summary


__all__ = ("derive_boundary_roles",)
