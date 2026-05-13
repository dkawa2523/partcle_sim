from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml

from .boundary_roles import derive_boundary_roles
from .compare_particle_results import write_field_alignment
from .export_requests import write_reextract_request_bundle
from .promotion import canonicalize_wall_event_table, is_wall_event_table
from .release_alignment import compare_release_tables


def _read_json_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _read_yaml_if_exists(path: Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML object expected: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, default=str) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=False), encoding="utf-8")


def _csv_shape(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False}
    frame = pd.read_csv(path, nrows=5)
    total = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace")) - 1
    return {
        "available": True,
        "path": str(path),
        "row_count": int(max(total, 0)),
        "columns": [str(col) for col in frame.columns],
    }


def _npz_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False}
    with np.load(path) as payload:
        keys = set(payload.files)
        metadata = {}
        if "metadata_json" in keys:
            raw = payload["metadata_json"]
            if isinstance(raw, np.ndarray):
                raw = raw.reshape(()).item() if raw.size == 1 else raw
            try:
                metadata = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw))
            except (TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
                metadata = {}
        quantities = sorted(
            key
            for key in keys
            if key
            not in {
                "axis_0",
                "axis_1",
                "axis_2",
                "times",
                "valid_mask",
                "metadata_json",
                "mesh_vertices",
                "mesh_triangles",
            }
        )
        out: dict[str, Any] = {
            "available": True,
            "path": str(path),
            "keys": sorted(keys),
            "quantities": quantities,
            "metadata": metadata,
        }
        if {"mesh_vertices", "mesh_triangles"} <= keys:
            out["backend"] = "triangle_mesh_2d"
            out["mesh_vertex_count"] = int(np.asarray(payload["mesh_vertices"]).shape[0])
            out["mesh_triangle_count"] = int(np.asarray(payload["mesh_triangles"]).shape[0])
        else:
            out["backend"] = "regular_rectilinear"
            if "valid_mask" in keys:
                out["grid_shape"] = [int(v) for v in np.asarray(payload["valid_mask"]).shape]
        if "times" in keys:
            times = np.asarray(payload["times"], dtype=np.float64)
            out["time_count"] = int(times.size)
            out["time_min_s"] = float(np.nanmin(times)) if times.size else None
            out["time_max_s"] = float(np.nanmax(times)) if times.size else None
        return out


def _features(raw_export_dir: Path) -> list[Mapping[str, Any]]:
    payload = _read_json_if_exists(raw_export_dir / "physics_feature_inventory.json")
    features = payload.get("features", [])
    return [feature for feature in features if isinstance(feature, Mapping)] if isinstance(features, list) else []


def _particle_release_features(raw_export_dir: Path) -> list[Mapping[str, Any]]:
    payload = _read_json_if_exists(raw_export_dir / "particle_release_inventory.json")
    features = payload.get("features", [])
    return [feature for feature in features if isinstance(feature, Mapping)] if isinstance(features, list) else []


def _is_particle_feature(feature: Mapping[str, Any]) -> bool:
    tag = str(feature.get("physics_tag", "")).strip().lower()
    label = str(feature.get("physics_label", "")).strip().lower()
    ptype = str(feature.get("physics_type", "")).strip().lower()
    return tag.startswith(("fpt", "pt")) or "particle tracing" in label or "particletracing" in ptype.replace(" ", "")


def _feature_values(feature: Mapping[str, Any]) -> Mapping[str, Any]:
    values = feature.get("property_values", {})
    if isinstance(values, Mapping):
        return values
    values = feature.get("known_settings", {})
    return values if isinstance(values, Mapping) else {}


def _scan_time_dependent_nonparticle_features(features: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    found = []
    time_expr = re.compile(r"(^|[^A-Za-z0-9_])t([^A-Za-z0-9_]|$)")
    for feature in features:
        if _is_particle_feature(feature):
            continue
        values = _feature_values(feature)
        for key, value in values.items():
            text = str(value)
            if time_expr.search(text):
                found.append(
                    {
                        "physics_tag": str(feature.get("physics_tag", "")),
                        "feature_tag": str(feature.get("feature_tag", "")),
                        "feature_type": str(feature.get("type", "")),
                        "property": str(key),
                        "expression": text,
                    }
                )
    return found


def _drag_force_inventory(features: Iterable[Mapping[str, Any]], run_config: Mapping[str, Any]) -> dict[str, Any]:
    drag_features = []
    for feature in features:
        if not _is_particle_feature(feature):
            continue
        if str(feature.get("type", "")).strip().lower() != "dragforce":
            continue
        values = dict(_feature_values(feature))
        drag_features.append(
            {
                "feature_tag": str(feature.get("feature_tag", "")),
                "feature_label": str(feature.get("label", "")),
                "drag_law": str(values.get("DragLaw", "")),
                "rarefaction_effects": str(values.get("Rarefaction_Effects", "")),
                "include_wall_corrections": str(values.get("IncludeWallCorrections", "")),
                "include_virtual_mass_pressure_gradient": str(values.get("IncludeVirtualMassAndPressureGradientForces", "")),
                "u_src": str(values.get("u_src", "")),
                "mu_source": str(values.get("mu_mat", values.get("mu", ""))),
                "rho_source": str(values.get("rho_mat", values.get("rho", ""))),
                "settings": values,
            }
        )
    solver_cfg = run_config.get("solver", {}) if isinstance(run_config.get("solver", {}), Mapping) else {}
    solver_forces = solver_cfg.get("forces", {}) if isinstance(solver_cfg.get("forces", {}), Mapping) else {}
    drag_cfg = solver_forces.get("drag", {}) if isinstance(solver_forces.get("drag", {}), Mapping) else {}
    solver_drag_model = str(drag_cfg.get("model", solver_cfg.get("drag_model", "stokes"))).strip().lower()

    def _force_enabled(name: str) -> bool:
        cfg = solver_forces.get(name, {})
        if isinstance(cfg, Mapping):
            return str(cfg.get("enabled", False)).strip().lower() in {"1", "true", "on", "yes"}
        return str(cfg).strip().lower() in {"1", "true", "on", "yes"}
    parity_gaps = []
    for item in drag_features:
        rare = str(item.get("rarefaction_effects", "")).strip().lower()
        if "cunningham" in rare and solver_drag_model != "stokes_cunningham":
            parity_gaps.append("COMSOL uses Cunningham-Millikan-Davies rarefaction but solver config is not stokes_cunningham")
        if str(item.get("include_virtual_mass_pressure_gradient", "")).strip() in {"1", "true", "on", "yes"}:
            missing = [name for name in ("virtual_mass", "pressure_gradient") if not _force_enabled(name)]
            if missing:
                parity_gaps.append(
                    "COMSOL enables virtual mass/pressure-gradient forces; missing enabled solver contribution(s): "
                    + ", ".join(missing)
                )
    return {
        "comsol_drag_features": drag_features,
        "solver_drag_model": solver_drag_model,
        "recommended_solver_drag_model": "stokes_cunningham"
        if any("cunningham" in str(item.get("rarefaction_effects", "")).lower() for item in drag_features)
        else solver_drag_model,
        "parity_gaps": sorted(set(parity_gaps)),
    }


def _release_feature_summary(raw_export_dir: Path) -> dict[str, Any]:
    features = _particle_release_features(raw_export_dir)
    release_rows = []
    property_rows = []
    for feature in features:
        values = dict(_feature_values(feature))
        row = {
            "feature_tag": str(feature.get("feature_tag", "")),
            "label": str(feature.get("label", "")),
            "type": str(feature.get("type", "")),
            "release_kind": str(feature.get("release_kind", "")),
            "selection_entities": list(feature.get("selection_entities", []) or []),
            "known_settings": dict(feature.get("known_settings", {}) if isinstance(feature.get("known_settings", {}), Mapping) else {}),
        }
        if str(feature.get("release_kind", "")) == "particle_properties":
            property_rows.append({**row, "property_values": values})
        elif str(feature.get("release_kind", "")) in {"release", "release_grid"}:
            release_rows.append(row)
    return {
        "feature_count": int(len(features)),
        "release_feature_count": int(len(release_rows)),
        "particle_property_feature_count": int(len(property_rows)),
        "release_features": release_rows,
        "particle_property_features": property_rows,
    }


def _wall_event_candidates(*paths: Path) -> list[str]:
    candidates = []
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            if is_wall_event_table(path):
                candidates.append(str(path))
    return sorted(candidates)


def _wall_event_truth_quality(*paths: Path) -> dict[str, Any]:
    candidates = _wall_event_candidates(*paths)
    details = []
    has_event_rows = False
    has_entity_values = False
    has_outcome_values = False
    has_normal_values = False
    for item in candidates:
        path = Path(item)
        try:
            frame = canonicalize_wall_event_table(path)
        except Exception as exc:  # noqa: BLE001 - audit reports candidate quality, not fatal scan errors
            details.append({"path": item, "status": "failed", "error": str(exc)})
            continue
        entity = pd.to_numeric(frame.get("comsol_entity_id", pd.Series(dtype=float)), errors="coerce").to_numpy(
            dtype=np.float64
        )
        normal_x = pd.to_numeric(frame.get("normal_x", pd.Series(dtype=float)), errors="coerce").to_numpy(
            dtype=np.float64
        )
        normal_y = pd.to_numeric(frame.get("normal_y", pd.Series(dtype=float)), errors="coerce").to_numpy(
            dtype=np.float64
        )
        outcome = frame.get("outcome", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
        entity_ok = bool(entity.size and np.isfinite(entity).any())
        outcome_ok = bool(len(outcome) and outcome.ne("").any())
        normal_ok = bool(normal_x.size and normal_y.size and np.isfinite(normal_x).any() and np.isfinite(normal_y).any())
        has_event_rows = has_event_rows or bool(len(frame))
        has_entity_values = has_entity_values or entity_ok
        has_outcome_values = has_outcome_values or outcome_ok
        has_normal_values = has_normal_values or normal_ok
        details.append(
            {
                "path": item,
                "status": "read",
                "row_count": int(len(frame)),
                "has_entity_values": entity_ok,
                "has_outcome_values": outcome_ok,
                "has_normal_values": normal_ok,
            }
        )
    return {
        "candidate_count": int(len(candidates)),
        "paths": candidates,
        "details": details,
        "has_event_rows": bool(has_event_rows),
        "has_entity_values": bool(has_entity_values),
        "has_outcome_values": bool(has_outcome_values),
        "has_normal_values": bool(has_normal_values),
        "direct_first_hit_entity_ready": bool(has_event_rows and has_entity_values),
    }


def _particle_status_truth_quality(*paths: Path) -> dict[str, Any]:
    candidates = []
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("comsol_particle_status.csv"):
            candidates.append(str(path))
    details = []
    has_rows = False
    has_stop_time_values = False
    has_final_status_values = False
    for item in sorted(candidates):
        path = Path(item)
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            details.append({"path": item, "status": "failed", "error": str(exc)})
            continue
        stop_col = _first_matching_column(frame.columns, ("stop_time_s", "status_stop_time_s", "fpt.st"))
        status_col = _first_matching_column(frame.columns, ("final_status", "status", "outcome"))
        stop_values = (
            pd.to_numeric(frame[stop_col], errors="coerce").to_numpy(dtype=np.float64)
            if stop_col is not None
            else np.asarray([], dtype=np.float64)
        )
        status_values = (
            frame[status_col].fillna("").astype(str).str.strip() if status_col is not None else pd.Series(dtype=str)
        )
        stop_ok = bool(stop_values.size and np.isfinite(stop_values).any())
        status_ok = bool(len(status_values) and status_values.ne("").any())
        has_rows = has_rows or bool(len(frame))
        has_stop_time_values = has_stop_time_values or stop_ok
        has_final_status_values = has_final_status_values or status_ok
        details.append(
            {
                "path": item,
                "status": "read",
                "row_count": int(len(frame)),
                "has_stop_time_values": stop_ok,
                "has_final_status_values": status_ok,
            }
        )
    return {
        "candidate_count": int(len(candidates)),
        "paths": sorted(candidates),
        "details": details,
        "has_status_rows": bool(has_rows),
        "has_stop_time_values": bool(has_stop_time_values),
        "has_final_status_values": bool(has_final_status_values),
        "status_stop_time_ready": bool(has_rows and has_stop_time_values),
        "interpretation": "fpt.st/fpt.fs particle status truth; not direct wall-hit entity/normal truth.",
    }


_RELEASE_SEMANTICS: dict[str, tuple[tuple[str, ...], ...]] = {
    "particle_id": (("particle_id", "ParticleID", "id", "pid", "particle"),),
    "release_time": (("release_time", "release_time_s", "t_release", "t0", "time_s", "time"),),
    "position": (
        ("x", "x_m", "x0", "x0_m", "r", "r_m", "r0", "r0_m"),
        ("y", "y_m", "y0", "y0_m", "z", "z_m", "z0", "z0_m"),
    ),
    "velocity": (
        ("v_x", "vx", "v_x0", "vx0", "vr", "vr0"),
        ("v_y", "vy", "v_y0", "vy0", "vz", "vz0"),
    ),
    "source": (("source_part_id", "source_entity", "source_boundary_id", "boundary_id", "part_id"),),
    "mass": (("mass", "mass_kg", "mp"),),
    "diameter": (("diameter", "diameter_m", "dp", "d"),),
    "density": (("density", "density_kgm3", "rho_p", "rhop"),),
    "charge": (("charge", "charge_C", "q"),),
}


def _first_matching_column(columns: Iterable[str], aliases: Iterable[str]) -> str | None:
    lower = {str(col).strip().lower(): str(col) for col in columns}
    for alias in aliases:
        found = lower.get(str(alias).strip().lower())
        if found is not None:
            return found
    return None


def _release_table_contract(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False}
    preview = pd.read_csv(path, nrows=5)
    frame = pd.read_csv(path)
    columns = [str(col) for col in frame.columns]
    semantics: dict[str, Any] = {}
    missing: list[str] = []
    for semantic, groups in _RELEASE_SEMANTICS.items():
        matched = []
        complete = True
        for aliases in groups:
            col = _first_matching_column(columns, aliases)
            matched.append(col or "")
            if col is None:
                complete = False
            elif not _column_has_value(frame, col):
                complete = False
        semantics[semantic] = {
            "present": bool(complete),
            "columns": matched,
        }
        if not complete:
            missing.append(semantic)
    required_for_exact_parity = list(_RELEASE_SEMANTICS.keys())
    return {
        "available": True,
        "path": str(path),
        "columns": [str(col) for col in preview.columns],
        "semantics": semantics,
        "required_for_exact_parity": required_for_exact_parity,
        "missing_for_exact_parity": missing,
        "exact_parity_ready": not missing,
    }


def _column_has_value(frame: pd.DataFrame, column: str) -> bool:
    series = frame[column]
    numeric = pd.to_numeric(series, errors="coerce")
    if np.isfinite(numeric.to_numpy(dtype=np.float64)).any():
        return True
    return bool(series.fillna("").astype(str).str.strip().ne("").any())


def _required_coordinate_scale(field_manifest: Mapping[str, Any]) -> float:
    if "coordinate_scale_m_per_model_unit" not in field_manifest:
        raise ValueError(
            "coordinate_scale_m_per_model_unit is required for COMSOL truth audit; "
            "implicit 1.0 scale is not allowed"
        )
    try:
        scale = float(field_manifest.get("coordinate_scale_m_per_model_unit"))
    except (TypeError, ValueError):
        scale = float("nan")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("coordinate_scale_m_per_model_unit must be a positive finite value")
    return scale


def _preferred_release_csv(particle_dir: Path) -> Path:
    for candidate in (
        particle_dir / "comsol_release_particles_canonical.csv",
        particle_dir / "canonical" / "comsol_release_particles_canonical.csv",
        particle_dir / "comsol_release_particles.csv",
    ):
        if candidate.exists():
            return candidate
    return particle_dir / "comsol_release_particles.csv"


def _finite_max(summary: Mapping[str, Any], section: str) -> float | None:
    value = summary.get(section, {}) if isinstance(summary, Mapping) else {}
    if not isinstance(value, Mapping):
        return None
    raw = value.get("max")
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _parity_readiness(
    *,
    missing_exports: list[str],
    force_parity_gaps: list[str],
    field_alignment: Mapping[str, Any],
    boundary_roles: Mapping[str, Any],
    release_exact: Mapping[str, Any],
    comsol_release_contract: Mapping[str, Any],
) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []
    for item in missing_exports:
        blockers.append(
            {
                "category": "missing_comsol_export",
                "severity": "blocker",
                "message": str(item),
            }
        )
    for item in force_parity_gaps:
        blockers.append(
            {
                "category": "force_parity",
                "severity": "blocker",
                "message": str(item),
            }
        )
    if bool(field_alignment.get("available", False)):
        support = field_alignment.get("support_fraction")
        try:
            support_value = float(support)
        except (TypeError, ValueError):
            support_value = float("nan")
        if not np.isfinite(support_value) or support_value < 1.0:
            blockers.append(
                {
                    "category": "field_support",
                    "severity": "blocker",
                    "message": "mesh field replay does not cover all COMSOL trajectory samples",
                    "support_fraction": support,
                }
            )
    else:
        blockers.append(
            {
                "category": "field_support",
                "severity": "blocker",
                "message": "mesh field replay was not executed",
            }
        )
    mismatch_count = boundary_roles.get("mismatch_count") if isinstance(boundary_roles, Mapping) else None
    if mismatch_count not in (0, None):
        blockers.append(
            {
                "category": "boundary_roles",
                "severity": "blocker",
                "message": "solver wall laws do not match COMSOL boundary role inventory",
                "mismatch_count": mismatch_count,
            }
        )
    if not bool(comsol_release_contract.get("exact_parity_ready", False)):
        blockers.append(
            {
                "category": "release_table_contract",
                "severity": "blocker",
                "message": "COMSOL release table lacks row-level exact-parity columns",
                "missing": list(comsol_release_contract.get("missing_for_exact_parity", [])),
            }
        )
    if not release_exact or release_exact.get("available") is False:
        blockers.append(
            {
                "category": "release_alignment",
                "severity": "blocker",
                "message": "exact release alignment was not computed",
            }
        )
    else:
        for section in ("release_time_error_s", "release_position_error_m", "release_velocity_error_mps"):
            max_error = _finite_max(release_exact, section)
            if max_error is not None and max_error > 0.0:
                blockers.append(
                    {
                        "category": "release_alignment",
                        "severity": "blocker",
                        "message": f"exact release {section} is nonzero",
                        "max": max_error,
                    }
                )
    return {
        "ready_for_exact_solver_comparison": not blockers,
        "blocker_count": int(len(blockers)),
        "blockers": blockers,
        "next_action_order": [
            "Run required COMSOL re-extraction configs",
            "Promote successful release/source/property probes into canonical release truth",
            "Promote successful wall-event probes into boundary-event comparison truth",
            "Re-run audit, then run exact-release solver comparison",
        ],
    }


def _root_cause_ranking(
    *,
    missing_exports: list[str],
    force_parity_gaps: list[str],
    field_alignment: Mapping[str, Any],
    field_truth: Mapping[str, Any],
    needs_time_resolved_field: bool,
    release_exact: Mapping[str, Any],
    release_clean: Mapping[str, Any],
    exact_input_contract: Mapping[str, Any],
    current_comparison: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if missing_exports:
        rows.append(
            {
                "category": "missing_comsol_truth",
                "status": "blocker",
                "finding": "COMSOL truth artifacts are still incomplete, so exact solver parity must not be claimed.",
                "evidence": list(missing_exports),
                "next_action": (
                    "Do not rerun the same fpt.st/fpt.fs probes. Use a different COMSOL event-dataset/API "
                    "path only if hit entity/normal truth is required."
                ),
            }
        )
    if force_parity_gaps:
        rows.append(
            {
                "category": "force_model",
                "status": "blocker",
                "finding": "The solver run config does not yet match the COMSOL Drag Force inventory.",
                "evidence": list(force_parity_gaps),
                "next_action": "Enable and re-baseline the matching solver force contributions separately from drag tuning.",
            }
        )
    exact_passed = exact_input_contract.get("passed") if isinstance(exact_input_contract, Mapping) else None
    status_counts = exact_input_contract.get("status_counts", {}) if isinstance(exact_input_contract, Mapping) else {}
    non_clean = int(status_counts.get("non_clean", 0) or 0) if isinstance(status_counts, Mapping) else 0
    if exact_passed is False or non_clean > 0:
        rows.append(
            {
                "category": "initial_support",
                "status": "debug_required",
                "finding": "Exact COMSOL release aligns row-for-row but violates the solver's strict initial field-support contract.",
                "evidence": {
                    "release_position_error_max_m": release_exact.get("release_position_error_m", {}).get("max")
                    if isinstance(release_exact, Mapping)
                    else None,
                    "input_status_counts": status_counts,
                    "non_clean_near_boundary_count": exact_input_contract.get("non_clean_near_boundary_count"),
                    "geometry_inside_violation_count": exact_input_contract.get("geometry_inside_violation_count"),
                },
                "next_action": "Keep exact release fixed as truth; classify mixed/boundary samples as solver contract gaps rather than moving particles.",
            }
        )
    if isinstance(current_comparison, Mapping) and current_comparison:
        trend = current_comparison.get("trend_alignment", {})
        divergence = current_comparison.get("divergence_alignment", {})
        rows.append(
            {
                "category": "boundary_event",
                "status": "debug_required",
                "finding": "Remaining trajectory divergence is concentrated near or after solver wall events.",
                "evidence": {
                    "solver_state_counts": trend.get("solver_state_counts", {}) if isinstance(trend, Mapping) else {},
                    "divergence_alignment": divergence.get("by_threshold", {}) if isinstance(divergence, Mapping) else {},
                    "first_hit_boundary_match_ratio": current_comparison.get("first_hit_boundary_match_ratio"),
                    "first_hit_time_comparison_count": current_comparison.get("first_hit_time_comparison_count"),
                    "first_hit_time_error_s": current_comparison.get("hit_time_error_s", {}),
                    "particle_status_stop_time_comparison_count": current_comparison.get(
                        "particle_status_stop_time_comparison_count"
                    ),
                    "solver_first_wall_vs_comsol_stop_time_error_s": current_comparison.get(
                        "solver_first_wall_vs_comsol_stop_time_error_s",
                        {},
                    ),
                },
                "next_action": (
                    "Treat fpt.st/fpt.fs as stop-time/final-status only. Direct first-hit entity/normal remains unavailable."
                ),
            }
        )
    if bool(field_alignment.get("available", False)):
        time_count = int(field_truth.get("time_count", 0) or 0)
        if time_count > 1:
            field_status = "partial_pass"
            field_finding = "Mesh-native time-resolved field replay covers all exported COMSOL trajectory samples."
            field_next_action = "Use the replay residuals by source and time to isolate interpolation and force-model effects."
        elif needs_time_resolved_field:
            field_status = "debug_required"
            field_finding = "Mesh-native field replay covers the COMSOL trajectory, but time-resolved field truth is still missing."
            field_next_action = "Promote time-resolved mesh field export, then compare steady-vs-transient replay residuals by source and time."
        else:
            field_status = "partial_pass"
            field_finding = "Mesh-native steady field replay covers the exported COMSOL trajectory samples."
            field_next_action = "Use residuals by source and time to isolate interpolation and force-model effects."
        rows.append(
            {
                "category": "field_interpolation",
                "status": field_status,
                "finding": field_finding,
                "evidence": {
                    "support_fraction": field_alignment.get("support_fraction"),
                    "field_time_count": time_count,
                    "field_time_min_s": field_truth.get("time_min_s"),
                    "field_time_max_s": field_truth.get("time_max_s"),
                    "velocity_residual_mps": field_alignment.get("velocity_residual_mps", {}),
                },
                "next_action": field_next_action,
            }
        )
    if release_clean and release_clean.get("available") is not False:
        rows.append(
            {
                "category": "diagnostic_release",
                "status": "not_truth",
                "finding": "The inward-clean release remains useful only for solver support diagnostics.",
                "evidence": {
                    "release_position_error_m": release_clean.get("release_position_error_m", {}),
                },
                "next_action": "Never use inward-clean particles as COMSOL exact release truth.",
            }
        )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = int(rank)
    return rows


def _solver_improvement_backlog(
    *,
    missing_exports: list[str],
    force_parity_gaps: list[str],
    field_alignment: Mapping[str, Any],
    field_truth: Mapping[str, Any],
    release_exact: Mapping[str, Any],
    exact_input_contract: Mapping[str, Any],
    current_comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract solver-code work from the audit without COMSOL/setup blockers."""

    items: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    if missing_exports:
        excluded.append(
            {
                "category": "missing_comsol_truth",
                "reason": "COMSOL evidence gap; keep out of solver-code backlog until direct truth exists.",
                "evidence": list(missing_exports),
            }
        )

    force_enablement_gaps = [
        gap
        for gap in force_parity_gaps
        if "no equivalent force contribution" in str(gap).lower()
        or "unsupported" in str(gap).lower()
        or "force_not_enabled" in str(gap).lower()
        or "virtual mass" in str(gap).lower()
        or "pressure-gradient" in str(gap).lower()
    ]
    config_force_gaps = [gap for gap in force_parity_gaps if gap not in force_enablement_gaps]
    if force_enablement_gaps:
        items.append(
            {
                "priority": "P0",
                "category": "force_model",
                "solver_area": "particle_tracer_unified/solvers/forces",
                "issue": "COMSOL force inventory includes contributions that must be enabled and re-baselined for this case.",
                "evidence": force_enablement_gaps,
                "current_solver_code_status": (
                    "Virtual mass and pressure-gradient contributions exist in the solver; this audit is tracking case configuration and baseline status."
                ),
                "code_evidence": [
                    {
                        "path": "particle_tracer_unified/solvers/forces/registry.py",
                        "symbols": ["ForceCatalog"],
                        "finding": "Contribution is recognized in the force catalog and can be reported as enabled or disabled.",
                    },
                    {
                        "path": "particle_tracer_unified/solvers/forces/runtime.py",
                        "symbols": ["ForceRuntimeParameters"],
                        "finding": "Runtime parameters expose virtual-mass and pressure-gradient execution settings.",
                    },
                ],
                "recommended_code_change": (
                    "Re-run the case with explicit force settings and compare acceleration contributions against COMSOL exports."
                ),
                "avoid": "Do not absorb this difference by changing drag coefficients, wall laws, or release positions.",
            }
        )
    if config_force_gaps:
        excluded.append(
            {
                "category": "force_configuration",
                "reason": "Solver already has a selectable model or config mismatch; not counted as core code debt here.",
                "evidence": config_force_gaps,
            }
        )

    exact_passed = exact_input_contract.get("passed") if isinstance(exact_input_contract, Mapping) else None
    status_counts = exact_input_contract.get("status_counts", {}) if isinstance(exact_input_contract, Mapping) else {}
    non_clean = int(status_counts.get("non_clean", 0) or 0) if isinstance(status_counts, Mapping) else 0
    if exact_passed is False or non_clean > 0:
        items.append(
            {
                "priority": "P1",
                "category": "initial_support",
                "solver_area": "particle_tracer_unified/core/field_backend.py",
                "issue": "Exact COMSOL boundary releases are valid truth inputs but violate the solver's strict clean-stencil contract.",
                "evidence": {
                    "release_position_error_max_m": release_exact.get("release_position_error_m", {}).get("max")
                    if isinstance(release_exact, Mapping)
                    else None,
                    "input_status_counts": status_counts,
                    "non_clean_near_boundary_count": exact_input_contract.get("non_clean_near_boundary_count"),
                    "geometry_inside_violation_count": exact_input_contract.get("geometry_inside_violation_count"),
                },
                "code_evidence": [
                    {
                        "path": "particle_tracer_unified/core/field_backend.py",
                        "symbols": ["sample_field_valid_status", "FieldSample"],
                        "finding": "Field support is classified before integration, but boundary-on-release is not a separate accepted state.",
                    },
                    {
                        "path": "particle_tracer_unified/core/triangle_mesh_sampling_2d.py",
                        "symbols": ["sample_triangle_mesh_status", "locate_triangle_containing_point"],
                        "finding": "Triangle mesh status is clean or hard-invalid; regular-grid mixed-stencil semantics do not map cleanly to exact boundary releases.",
                    },
                ],
                "recommended_code_change": (
                    "Make boundary-on-release handling an explicit solver contract: accept mesh-native boundary samples "
                    "with a separate status or fail deterministically before integration."
                ),
                "avoid": "Do not move release particles or use geometry-mask/ghost-cell padding as faithful truth.",
            }
        )

    if isinstance(current_comparison, Mapping) and current_comparison:
        trend = current_comparison.get("trend_alignment", {})
        divergence = current_comparison.get("divergence_alignment", {})
        stop_error = current_comparison.get("solver_first_wall_vs_comsol_stop_time_error_s", {})
        state_counts = trend.get("solver_state_counts", {}) if isinstance(trend, Mapping) else {}
        if state_counts or stop_error:
            items.append(
                {
                    "priority": "P1",
                    "category": "boundary_contact_state",
                    "solver_area": "particle_tracer_unified/solvers/high_fidelity_collision.py",
                    "issue": "Solver contact outcomes are dominated by endpoint/sliding states, and stop-time residuals remain large.",
                    "evidence": {
                        "solver_state_counts": state_counts,
                        "solver_first_wall_vs_comsol_stop_time_error_s": stop_error,
                        "divergence_alignment": divergence.get("by_threshold", {}) if isinstance(divergence, Mapping) else {},
                    },
                    "code_evidence": [
                        {
                            "path": "particle_tracer_unified/solvers/high_fidelity_collision.py",
                            "symbols": ["_apply_wall_hit_step", "_same_wall_contact_sliding_state"],
                            "finding": "Repeated hits can be converted into persistent contact/sliding states.",
                        },
                        {
                            "path": "particle_tracer_unified/solvers/runtime_outputs.py",
                            "symbols": ["_particle_state_labels", "_build_collision_diag_report"],
                            "finding": "Final diagnostics expose contact_sliding/contact_endpoint_stopped as solver states.",
                        },
                        {
                            "path": "particle_tracer_unified/core/boundary_hits.py",
                            "symbols": ["segment_hit_from_boundary_edges", "_is_pass_through_part"],
                            "finding": "Pass-through surfaces are filtered correctly; remaining divergence is not PairContinuity mapping.",
                        },
                    ],
                    "recommended_code_change": (
                        "Simplify boundary contact into explicit pass-through, stick/freeze, specular bounce, and "
                        "well-defined endpoint handling; emit first-contact diagnostics from one state machine."
                    ),
                    "avoid": "Do not infer COMSOL first-hit entity from particle stop-time/status columns.",
                }
            )

    if bool(field_alignment.get("available", False)):
        velocity = field_alignment.get("velocity_residual_mps", {})
        p99 = velocity.get("p99") if isinstance(velocity, Mapping) else None
        try:
            p99_value = float(p99)
        except (TypeError, ValueError):
            p99_value = float("nan")
        if np.isfinite(p99_value) and p99_value > 1.0e-3:
            items.append(
                {
                    "priority": "P2",
                    "category": "field_interpolation",
                    "solver_area": "particle_tracer_unified/core/triangle_mesh_sampling_2d.py",
                    "issue": "Mesh-native field replay has full support but nontrivial velocity residuals near trajectory samples.",
                    "evidence": {
                        "support_fraction": field_alignment.get("support_fraction"),
                        "field_time_count": field_truth.get("time_count"),
                        "velocity_residual_mps": velocity,
                    },
                    "code_evidence": [
                        {
                            "path": "particle_tracer_unified/core/triangle_mesh_sampling_2d.py",
                            "symbols": ["sample_triangle_mesh_series"],
                            "finding": "Replay uses barycentric spatial interpolation plus linear time interpolation.",
                        },
                        {
                            "path": "particle_tracer_unified/core/field_backend.py",
                            "symbols": ["sample_field_quantity"],
                            "finding": "Mesh-native and regular-grid sampling share the public field API, so residual isolation belongs below this boundary.",
                        },
                    ],
                    "recommended_code_change": (
                        "After force/contact fixes, isolate interpolation residual by source/time and verify element lookup, "
                        "time interpolation, and unit handling."
                    ),
                    "avoid": "Do not switch back to regular-grid or ghost-cell truth for boundary-near replay.",
                }
            )

    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    items.sort(key=lambda item: (priority_order.get(str(item.get("priority")), 99), str(item.get("category", ""))))
    for rank, item in enumerate(items, start=1):
        item["rank"] = int(rank)
    return {
        "scope": "solver_code_only_excluding_comsol_setup_and_missing_truth",
        "conclusion": "solver_code_changes_required" if items else "no_solver_code_backlog_from_current_audit",
        "item_count": int(len(items)),
        "items": items,
        "excluded_from_solver_code_backlog": excluded,
    }


def build_truth_audit(
    *,
    case_name: str,
    field_raw_dir: str | Path,
    particle_raw_dir: str | Path | None = None,
    solver_case_dir: str | Path | None = None,
    out_dir: str | Path,
    field_npz: str | Path | None = None,
    regular_field_npz: str | Path | None = None,
    solver_output_dir: str | Path | None = None,
    comparison_dir: str | Path | None = None,
    run_config: str | Path | None = None,
    compare_field_replay: bool = True,
) -> dict[str, Any]:
    field_dir = Path(field_raw_dir)
    particle_dir = Path(particle_raw_dir) if particle_raw_dir is not None else field_dir
    case_dir = Path(solver_case_dir) if solver_case_dir is not None else None
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    field_manifest = _read_json_if_exists(field_dir / "export_manifest.json")
    particle_manifest = _read_json_if_exists(particle_dir / "export_manifest.json")
    _required_coordinate_scale(field_manifest)
    trajectory_report = _read_json_if_exists(particle_dir / "comsol_particle_trajectory_report.json")
    exact_input_contract = (
        _read_json_if_exists(case_dir / "generated" / "exact_compare_input_contract_summary.json")
        if case_dir is not None
        else {}
    )
    comparison_path = (
        Path(comparison_dir) / "comparison_summary.json"
        if comparison_dir is not None
        else particle_dir / "solver_inward_clean_mesh_tol25um_disttol_trend" / "comparison_summary.json"
    )
    current_comparison = _read_json_if_exists(comparison_path)
    features = _features(field_dir)
    run_cfg_path = Path(run_config) if run_config is not None else (case_dir / "run_config_inward_clean_trend_mesh.yaml" if case_dir is not None else None)
    run_cfg = _read_yaml_if_exists(run_cfg_path)

    field_npz_path = Path(field_npz) if field_npz is not None else (case_dir / "generated" / "comsol_field_mesh_2d.npz" if case_dir is not None else None)
    regular_npz_path = Path(regular_field_npz) if regular_field_npz is not None else (case_dir / "generated" / "comsol_field_2d.npz" if case_dir is not None else None)
    field_npz_summary = _npz_summary(field_npz_path)
    boundary_map_csv = case_dir / "generated" / "comsol_boundary_entity_mapping.csv" if case_dir is not None else None
    part_walls_csv = case_dir / "part_walls.csv" if case_dir is not None else None
    exact_particles_csv = case_dir / "particles.csv" if case_dir is not None else None
    clean_particles_csv = case_dir / "particles_inward_clean.csv" if case_dir is not None else None
    comsol_release_csv = _preferred_release_csv(particle_dir)
    comsol_trajectory_csv = particle_dir / "comsol_particle_trajectory.csv"

    release_exact = {"available": False}
    if exact_particles_csv is not None and exact_particles_csv.exists() and comsol_release_csv.exists():
        release_exact = compare_release_tables(exact_particles_csv, comsol_release_csv, out_dir=output_dir / "release_exact")
    release_clean = {"available": False}
    if clean_particles_csv is not None and clean_particles_csv.exists() and comsol_release_csv.exists():
        release_clean = compare_release_tables(clean_particles_csv, comsol_release_csv, out_dir=output_dir / "release_inward_clean")

    boundary_roles = {"available": False}
    if boundary_map_csv is not None and boundary_map_csv.exists():
        boundary_roles = derive_boundary_roles(
            raw_export_dir=field_dir,
            boundary_map_csv=boundary_map_csv,
            part_walls_csv=part_walls_csv if part_walls_csv is not None and part_walls_csv.exists() else None,
            out_dir=output_dir / "boundary_roles",
        )

    field_alignment = {"available": False}
    if compare_field_replay and field_npz_path is not None and field_npz_path.exists() and comsol_trajectory_csv.exists():
        field_alignment = write_field_alignment(
            field_npz=field_npz_path,
            comsol_trajectory_csv=comsol_trajectory_csv,
            solver_particles_csv=clean_particles_csv if clean_particles_csv is not None and clean_particles_csv.exists() else exact_particles_csv,
            out_dir=output_dir / "field_replay_mesh",
        )

    time_dependent_features = _scan_time_dependent_nonparticle_features(features)
    field_context_count = int(field_manifest.get("field_sample_context_count", 0) or 0)
    field_truth_time_count = max(field_context_count, int(field_npz_summary.get("time_count", 0) or 0))
    needs_time_resolved_field = bool(time_dependent_features and field_truth_time_count <= 1)
    comsol_release_contract = _release_table_contract(comsol_release_csv)
    exact_release_contract = _release_table_contract(exact_particles_csv)
    clean_release_contract = _release_table_contract(clean_particles_csv)
    missing_exports = []
    if needs_time_resolved_field:
        missing_exports.append("time-resolved mesh field export is required before claiming time-varying field parity")
    wall_event_truth = _wall_event_truth_quality(field_dir, particle_dir)
    particle_status_truth = _particle_status_truth_quality(field_dir, particle_dir)
    if not particle_status_truth["status_stop_time_ready"] and not wall_event_truth["has_event_rows"]:
        missing_exports.append("COMSOL particle status/stop-time export is missing; boundary-event parity cannot be direct")
    if not wall_event_truth["direct_first_hit_entity_ready"]:
        missing_exports.append(
            "COMSOL wall-hit entity/normal export is unavailable; fpt.st/fpt.fs only provide particle stop-time/status"
        )
    comsol_release_missing = list(comsol_release_contract.get("missing_for_exact_parity", []))
    if comsol_release_missing:
        missing_exports.append(
            "COMSOL release table is missing row-level exact-parity columns: "
            + ", ".join(str(item) for item in comsol_release_missing)
        )

    reextract_requests = write_reextract_request_bundle(
        case_name=str(case_name),
        field_manifest=field_manifest,
        particle_manifest=particle_manifest,
        trajectory_report=trajectory_report,
        trajectory_csv=comsol_trajectory_csv,
        out_dir=output_dir / "required_comsol_exports",
        needs_time_resolved_field=needs_time_resolved_field,
        needs_wall_events=not bool(particle_status_truth["status_stop_time_ready"])
        and not bool(wall_event_truth["direct_first_hit_entity_ready"]),
        needs_release_properties=bool(comsol_release_missing),
    )
    forces = _drag_force_inventory(features, run_cfg)
    parity_readiness = _parity_readiness(
        missing_exports=missing_exports,
        force_parity_gaps=list(forces.get("parity_gaps", [])),
        field_alignment=field_alignment,
        boundary_roles=boundary_roles,
        release_exact=release_exact,
        comsol_release_contract=comsol_release_contract,
    )
    root_cause_ranking = _root_cause_ranking(
        missing_exports=missing_exports,
        force_parity_gaps=list(forces.get("parity_gaps", [])),
        field_alignment=field_alignment,
        field_truth=field_npz_summary,
        needs_time_resolved_field=needs_time_resolved_field,
        release_exact=release_exact,
        release_clean=release_clean,
        exact_input_contract=exact_input_contract,
        current_comparison=current_comparison,
    )
    solver_improvement_backlog = _solver_improvement_backlog(
        missing_exports=missing_exports,
        force_parity_gaps=list(forces.get("parity_gaps", [])),
        field_alignment=field_alignment,
        field_truth=field_npz_summary,
        release_exact=release_exact,
        exact_input_contract=exact_input_contract,
        current_comparison=current_comparison,
    )

    manifest = {
        "case_name": str(case_name),
        "audit_inputs": {
            "field_raw_dir": str(field_dir),
            "particle_raw_dir": str(particle_dir),
            "solver_case_dir": str(case_dir) if case_dir is not None else "",
            "solver_output_dir": str(solver_output_dir) if solver_output_dir is not None else "",
            "comparison_dir": str(comparison_dir) if comparison_dir is not None else "",
            "run_config": str(run_cfg_path) if run_cfg_path is not None else "",
        },
        "comsol_model": {
            "mph_path": field_manifest.get("mph_path", particle_manifest.get("mph_path", "")),
            "mph_sha256": field_manifest.get("mph_sha256", particle_manifest.get("mph_sha256", "")),
            "version": trajectory_report.get("metadata", {}).get("Version", ""),
        },
        "coordinates": {
            "axis_names": field_manifest.get("axis_names", []),
            "coordinate_model_unit": field_manifest.get("coordinate_model_unit", ""),
            "coordinate_scale_m_per_model_unit": _required_coordinate_scale(field_manifest),
        },
        "datasets": {
            "field_dataset": field_manifest.get("dataset", ""),
            "particle_dataset": particle_manifest.get("data_export_dataset", "part1"),
            "mesh_tag": field_manifest.get("mesh_tag", ""),
            "field_sample_context_count": field_context_count,
            "particle_trajectory_time_count": trajectory_report.get("trajectory_time_count", None),
            "particle_trajectory_time_min_s": trajectory_report.get("time_min_s", None),
            "particle_trajectory_time_max_s": trajectory_report.get("time_max_s", None),
            "time_dependent_nonparticle_feature_count": int(len(time_dependent_features)),
        },
        "field_truth": {
            "preferred_backend": "triangle_mesh_2d",
            "mesh_field_npz": field_npz_summary,
            "regular_grid_npz": {
                **_npz_summary(regular_npz_path),
                "status": "diagnostic_only_boundary_near_replay_not_primary",
            },
            "field_alignment": field_alignment,
        },
        "forces": forces,
        "particles": {
            "release_inventory": _release_feature_summary(field_dir),
            "comsol_release_csv": _csv_shape(comsol_release_csv),
            "solver_exact_release_csv": _csv_shape(exact_particles_csv),
            "solver_inward_clean_release_csv": _csv_shape(clean_particles_csv),
            "exact_release_alignment": release_exact,
            "inward_clean_release_alignment": release_clean,
            "release_table_contract": {
                "comsol_release": comsol_release_contract,
                "solver_exact_release": exact_release_contract,
                "solver_inward_clean_release": clean_release_contract,
            },
            "interpretation": {
                "exact_release": "truth input for initial-condition parity",
                "inward_clean": "diagnostic input adjusted to satisfy solver field-support constraints",
            },
        },
        "boundaries": {
            "boundary_map_csv": _csv_shape(boundary_map_csv),
            "part_walls_csv": _csv_shape(part_walls_csv),
            "boundary_role_alignment": boundary_roles,
            "wall_event_export_candidates": wall_event_truth["paths"],
            "wall_event_truth_quality": wall_event_truth,
            "particle_status_truth_quality": particle_status_truth,
            "comparison_semantics": {
                "particle_status": "COMSOL fpt.st/fpt.fs stop-time/final-status truth",
                "wall_event": "direct wall-hit entity/normal truth; not satisfied by fpt.st/fpt.fs alone",
            },
        },
        "time_dependency_risk": {
            "time_dependent_nonparticle_features": time_dependent_features[:20],
            "truncated": bool(len(time_dependent_features) > 20),
        },
        "missing_comsol_exports": missing_exports,
        "comsol_reextract_requests": reextract_requests,
        "parity_readiness": parity_readiness,
        "root_cause_ranking": root_cause_ranking,
        "solver_improvement_backlog": solver_improvement_backlog,
        "cleanup_policy": {
            "regular_grid_is_primary_truth": False,
            "inward_clean_is_comsol_truth": False,
            "placeholder_particles_allowed": False,
            "placeholder_walls_allowed": False,
            "field_ghost_cells_allowed_for_faithful": False,
        },
    }

    summary = {
        "case_name": str(case_name),
        "truth_manifest_yaml": str(output_dir / "micromixer_truth_manifest.yaml"),
        "truth_manifest_json": str(output_dir / "micromixer_truth_manifest.json"),
        "missing_comsol_exports": missing_exports,
        "field_alignment_available": bool(field_alignment.get("available", False)),
        "field_support_fraction": field_alignment.get("support_fraction"),
        "boundary_role_mismatch_count": boundary_roles.get("mismatch_count") if isinstance(boundary_roles, Mapping) else None,
        "force_parity_gaps": manifest["forces"].get("parity_gaps", []),
        "release_contract_missing": comsol_release_missing,
        "ready_for_exact_solver_comparison": bool(parity_readiness.get("ready_for_exact_solver_comparison", False)),
        "parity_blocker_count": int(parity_readiness.get("blocker_count", 0)),
        "root_cause_ranking": root_cause_ranking,
        "solver_improvement_backlog": str(output_dir / "solver_improvement_backlog.json"),
        "solver_improvement_item_count": int(solver_improvement_backlog.get("item_count", 0)),
        "reextract_request_summary": str(output_dir / "required_comsol_exports" / "reextract_request_summary.json"),
        "exact_release_position_error_m": release_exact.get("release_position_error_m", {}),
        "inward_clean_release_position_error_m": release_clean.get("release_position_error_m", {}),
    }
    _write_yaml(output_dir / "micromixer_truth_manifest.yaml", manifest)
    _write_json(output_dir / "micromixer_truth_manifest.json", manifest)
    _write_json(output_dir / "micromixer_audit_summary.json", summary)
    _write_json(output_dir / "root_cause_ranking.json", {"case_name": str(case_name), "findings": root_cause_ranking})
    _write_json(
        output_dir / "solver_improvement_backlog.json",
        {"case_name": str(case_name), **solver_improvement_backlog},
    )
    return summary


__all__ = ("build_truth_audit",)
