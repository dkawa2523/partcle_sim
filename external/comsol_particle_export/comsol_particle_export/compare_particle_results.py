from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from particle_tracer_unified.core.field_sampling import VALID_MASK_STATUS_CLEAN
from particle_tracer_unified.core.triangle_mesh_sampling_2d import (
    sample_triangle_mesh_series,
    sample_triangle_mesh_status,
)
from particle_tracer_unified.providers.precomputed import build_precomputed_triangle_mesh_field

from .boundary_roles import derive_boundary_roles


SOLVER_STATE_ORDER = (
    "active_free_flight",
    "contact_sliding",
    "contact_endpoint_stopped",
    "invalid_mask_stopped",
    "numerical_boundary_stopped",
    "stuck",
    "absorbed",
    "escaped",
    "inactive",
)

COMSOL_STATE_MAP = {
    "active": "active_free_flight",
    "alive": "active_free_flight",
    "running": "active_free_flight",
    "free": "active_free_flight",
    "stuck": "stuck",
    "stick": "stuck",
    "sticking": "stuck",
    "deposited": "stuck",
    "deposition": "stuck",
    "frozen": "stuck",
    "freeze": "stuck",
    "attached": "stuck",
    "absorbed": "absorbed",
    "absorb": "absorbed",
    "disappeared": "absorbed",
    "disappear": "absorbed",
    "removed": "absorbed",
    "terminated": "absorbed",
    "escaped": "escaped",
    "escape": "escaped",
    "outlet": "escaped",
    "exit": "escaped",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _first_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lower = {str(c).strip().lower(): str(c) for c in frame.columns}
    for name in aliases:
        key = str(name).strip().lower()
        if key in lower:
            return lower[key]
    return None


def _numeric_or_nan(frame: pd.DataFrame, aliases: Iterable[str]) -> np.ndarray:
    col = _first_column(frame, aliases)
    if col is None:
        return np.full(len(frame), np.nan, dtype=np.float64)
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)


def _string_or_empty(frame: pd.DataFrame, aliases: Iterable[str]) -> np.ndarray:
    col = _first_column(frame, aliases)
    if col is None:
        return np.full(len(frame), "", dtype=object)
    return frame[col].fillna("").astype(str).to_numpy(dtype=object)


def _finite_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90.0)),
        "p99": float(np.percentile(finite, 99.0)),
        "max": float(np.max(finite)),
    }


def _bounds_summary(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for col in frame.columns:
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            out[str(col)] = {"min": float(np.min(finite)), "max": float(np.max(finite))}
    return out


def _solver_state_labels(final_particles: pd.DataFrame) -> np.ndarray:
    labels = np.full(len(final_particles), "inactive", dtype=object)
    for name in SOLVER_STATE_ORDER:
        if name in final_particles.columns:
            mask = pd.to_numeric(final_particles[name], errors="coerce").fillna(0).to_numpy(dtype=np.int64) != 0
            labels[mask] = name
    if "active" in final_particles.columns:
        mask = pd.to_numeric(final_particles["active"], errors="coerce").fillna(0).to_numpy(dtype=np.int64) != 0
        labels[mask] = "active_free_flight"
    for name in ("contact_sliding", "contact_endpoint_stopped", "invalid_mask_stopped", "numerical_boundary_stopped", "stuck", "absorbed", "escaped"):
        if name in final_particles.columns:
            mask = pd.to_numeric(final_particles[name], errors="coerce").fillna(0).to_numpy(dtype=np.int64) != 0
            labels[mask] = name
    return labels


def _normalize_comsol_state(values: np.ndarray, state_map: Mapping[str, str]) -> np.ndarray:
    out = []
    merged_map = dict(COMSOL_STATE_MAP)
    merged_map.update({str(k).strip().lower(): str(v) for k, v in state_map.items()})
    for raw in values:
        key = str(raw).strip().lower()
        out.append(merged_map.get(key, key if key else "unknown"))
    return np.asarray(out, dtype=object)


def _particle_id(frame: pd.DataFrame) -> np.ndarray:
    col = _first_column(frame, ("particle_id", "ParticleID", "id", "pid", "particle"))
    if col is None:
        raise ValueError("particle result table is missing a particle_id column")
    return pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=np.int64)


def _position_frame(frame: pd.DataFrame, *, prefix: str = "") -> pd.DataFrame:
    if prefix:
        aliases = {
            "x": (f"{prefix}_x_m", f"{prefix}_x", f"x_{prefix}_m", f"x_{prefix}"),
            "y": (f"{prefix}_y_m", f"{prefix}_y", f"y_{prefix}_m", f"y_{prefix}"),
            "z": (f"{prefix}_z_m", f"{prefix}_z", f"z_{prefix}_m", f"z_{prefix}"),
        }
    else:
        x_col = _first_column(frame, ("x_m", "x", "final_x_m", "final_x"))
        y_col = _first_column(frame, ("y_m", "y", "final_y_m", "final_y"))
        r_col = _first_column(frame, ("r_m", "r"))
        rz_col = _first_column(frame, ("z_m", "z"))
        if x_col is None and y_col is None and r_col is not None and rz_col is not None:
            return pd.DataFrame(
                {
                    "x": pd.to_numeric(frame[r_col], errors="coerce").to_numpy(dtype=np.float64),
                    "y": pd.to_numeric(frame[rz_col], errors="coerce").to_numpy(dtype=np.float64),
                }
            )
        aliases = {
            "x": ("x_m", "x", "final_x_m", "final_x"),
            "y": ("y_m", "y", "final_y_m", "final_y"),
            "z": ("z_m", "z", "final_z_m", "final_z"),
        }
    data: dict[str, np.ndarray] = {}
    for axis, names in aliases.items():
        values = _numeric_or_nan(frame, names)
        if np.isfinite(values).any():
            data[axis] = values
    return pd.DataFrame(data)


def _velocity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "v_x": ("v_x", "vx", "u", "final_vx", "v_x_mps"),
        "v_y": ("v_y", "vy", "v", "w", "final_vy", "v_y_mps"),
        "v_z": ("v_z", "vz", "final_vz", "v_z_mps"),
    }
    data: dict[str, np.ndarray] = {}
    for axis, names in aliases.items():
        values = _numeric_or_nan(frame, names)
        if np.isfinite(values).any():
            data[axis] = values
    return pd.DataFrame(data)


def _norm_error(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    common = [c for c in left.columns if c in right.columns]
    if not common:
        return np.full(min(len(left), len(right)), np.nan, dtype=np.float64)
    a = left[common].to_numpy(dtype=np.float64)
    b = right[common].to_numpy(dtype=np.float64)
    valid = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
    out = np.full(len(left), np.nan, dtype=np.float64)
    out[valid] = np.linalg.norm(a[valid] - b[valid], axis=1)
    return out


def _solver_final_frame(output_dir: Path) -> pd.DataFrame:
    final = _read_csv(output_dir / "final_particles.csv")
    out = pd.DataFrame({"particle_id": _particle_id(final), "solver_state": _solver_state_labels(final)})
    pos = _position_frame(final)
    vel = _velocity_frame(final)
    for col in pos.columns:
        out[f"solver_final_{col}"] = pos[col].to_numpy(dtype=np.float64)
    for col in vel.columns:
        out[f"solver_{col}"] = vel[col].to_numpy(dtype=np.float64)
    out["solver_charge_C"] = _numeric_or_nan(final, ("charge_C", "charge", "q"))
    return out


def _solver_first_hit_frame(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "wall_events.csv"
    if not path.exists():
        return pd.DataFrame(columns=["particle_id"])
    events = _read_csv(path)
    if events.empty:
        return pd.DataFrame(columns=["particle_id"])
    events = events.copy()
    events["particle_id"] = _particle_id(events)
    events["solver_hit_time_s"] = _numeric_or_nan(events, ("hit_time_s", "time_s", "t"))
    events = events.sort_values(["particle_id", "solver_hit_time_s"], na_position="last")
    first = events.groupby("particle_id", as_index=False).first()
    out = pd.DataFrame(
        {
            "particle_id": first["particle_id"].to_numpy(dtype=np.int64),
            "solver_hit_time_s": _numeric_or_nan(first, ("solver_hit_time_s", "hit_time_s", "time_s")),
            "solver_hit_part_id": _numeric_or_nan(first, ("part_id", "solver_part_id")).astype("float64"),
            "solver_hit_outcome": _string_or_empty(first, ("outcome", "wall_mode")),
            "solver_impact_speed_mps": _numeric_or_nan(first, ("impact_speed_mps",)),
        }
    )
    hit_pos = _position_frame(first.rename(columns={"hit_x_m": "hit_x", "hit_y_m": "hit_y", "hit_z_m": "hit_z"}), prefix="hit")
    for col in hit_pos.columns:
        out[f"solver_hit_{col}"] = hit_pos[col].to_numpy(dtype=np.float64)
    return out


def _load_boundary_map(path: Path | None) -> dict[int, int]:
    if path is None:
        return {}
    frame = _read_csv(path)
    left = _first_column(
        frame,
        (
            "comsol_boundary_id",
            "comsol_entity_id",
            "comsol_edge_entity_id",
            "comsol_api_selection_entity_id",
            "boundary_id",
            "hit_boundary_id",
        ),
    )
    right = _first_column(frame, ("solver_part_id", "part_id"))
    if left is None or right is None:
        raise ValueError("boundary map must contain comsol_boundary_id and solver_part_id columns")
    return {
        int(c): int(s)
        for c, s in zip(
            pd.to_numeric(frame[left], errors="coerce"),
            pd.to_numeric(frame[right], errors="coerce"),
        )
        if pd.notna(c) and pd.notna(s)
    }


def _comsol_frame(path: Path, *, boundary_map: Mapping[int, int], state_map: Mapping[str, str]) -> pd.DataFrame:
    raw = _read_csv(path)
    raw = raw.copy()
    raw["_particle_id_normalized"] = _particle_id(raw)
    time_col = _first_column(raw, ("time_s", "time", "t"))
    if raw["_particle_id_normalized"].duplicated().any():
        if time_col is not None:
            raw["_time_sort"] = pd.to_numeric(raw[time_col], errors="coerce")
            raw = raw.sort_values(["_particle_id_normalized", "_time_sort"], na_position="last")
        raw = raw.groupby("_particle_id_normalized", as_index=False).last()
    out = pd.DataFrame({"particle_id": _particle_id(raw)})
    states = _string_or_empty(raw, ("final_state", "particle_state", "state", "status", "outcome"))
    out["comsol_state"] = _normalize_comsol_state(states, state_map)
    pos = _position_frame(raw)
    vel = _velocity_frame(raw)
    for col in pos.columns:
        out[f"comsol_final_{col}"] = pos[col].to_numpy(dtype=np.float64)
    for col in vel.columns:
        out[f"comsol_{col}"] = vel[col].to_numpy(dtype=np.float64)
    out["comsol_charge_C"] = _numeric_or_nan(raw, ("charge_C", "charge", "q"))
    out["comsol_hit_time_s"] = _numeric_or_nan(raw, ("hit_time_s", "hit_time", "t_hit", "wall_time", "event_time_s"))
    raw_boundary = _numeric_or_nan(
        raw,
        (
            "comsol_entity_id",
            "entity_id",
            "hit_boundary_id",
            "boundary_id",
            "hit_boundary",
            "part_id",
        ),
    )
    mapped = []
    for value in raw_boundary:
        if np.isfinite(value):
            mapped.append(float(boundary_map.get(int(value), int(value))))
        else:
            mapped.append(np.nan)
    out["comsol_hit_part_id"] = np.asarray(mapped, dtype=np.float64)
    out["comsol_hit_outcome"] = _string_or_empty(raw, ("hit_outcome", "wall_outcome", "outcome", "wall_mode"))
    hit_pos = _position_frame(raw, prefix="hit")
    for col in hit_pos.columns:
        out[f"comsol_hit_{col}"] = hit_pos[col].to_numpy(dtype=np.float64)
    return out


def _comsol_wall_event_frame(path: Path, *, boundary_map: Mapping[int, int]) -> pd.DataFrame:
    raw = _read_csv(path)
    raw = raw.copy()
    raw["particle_id"] = _particle_id(raw)
    raw["_hit_time_sort"] = _numeric_or_nan(raw, ("hit_time_s", "hit_time", "t_hit", "wall_time", "event_time_s"))
    raw = raw.sort_values(["particle_id", "_hit_time_sort"], na_position="last")
    first = raw.groupby("particle_id", as_index=False).first()
    out = pd.DataFrame(
        {
            "particle_id": first["particle_id"].to_numpy(dtype=np.int64),
            "comsol_hit_time_s": _numeric_or_nan(first, ("hit_time_s", "hit_time", "t_hit", "wall_time", "event_time_s")),
            "comsol_hit_outcome": _string_or_empty(first, ("hit_outcome", "wall_outcome", "outcome", "wall_mode", "status")),
        }
    )
    raw_boundary = _numeric_or_nan(
        first,
        (
            "comsol_entity_id",
            "entity_id",
            "hit_boundary_id",
            "boundary_id",
            "hit_boundary",
            "part_id",
        ),
    )
    mapped = []
    for value in raw_boundary:
        if np.isfinite(value):
            mapped.append(float(boundary_map.get(int(value), int(value))))
        else:
            mapped.append(np.nan)
    out["comsol_hit_part_id"] = np.asarray(mapped, dtype=np.float64)
    hit_pos = _position_frame(first, prefix="hit")
    for col in hit_pos.columns:
        out[f"comsol_hit_{col}"] = hit_pos[col].to_numpy(dtype=np.float64)
    return out


def _comsol_particle_status_frame(path: Path) -> pd.DataFrame:
    raw = _read_csv(path)
    raw = raw.copy()
    out = pd.DataFrame(
        {
            "particle_id": _particle_id(raw),
            "comsol_status_stop_time_s": _numeric_or_nan(raw, ("stop_time_s", "status_stop_time_s", "fpt.st")),
            "comsol_final_status": _string_or_empty(raw, ("final_status", "status", "outcome")),
            "comsol_final_status_code": _numeric_or_nan(raw, ("final_status_code", "status_code", "fpt.fs")),
        }
    )
    return out.sort_values(["particle_id", "comsol_status_stop_time_s"], na_position="last").groupby(
        "particle_id", as_index=False
    ).first()


def _merge_comsol_wall_events(comsol: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return comsol
    merged = comsol.merge(events, on="particle_id", how="left", suffixes=("", "_event"))
    for col in events.columns:
        if col == "particle_id":
            continue
        event_col = f"{col}_event"
        if event_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = merged[event_col]
            merged = merged.drop(columns=[event_col])
            continue
        if col.endswith("_outcome"):
            event_values = merged[event_col].fillna("").astype(str)
            base_values = merged[col].fillna("").astype(str)
            merged[col] = np.where(event_values.str.strip().ne("").to_numpy(), event_values.to_numpy(), base_values.to_numpy())
        else:
            event_values = pd.to_numeric(merged[event_col], errors="coerce").to_numpy(dtype=np.float64)
            base_values = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=np.float64)
            merged[col] = np.where(np.isfinite(event_values), event_values, base_values)
        merged = merged.drop(columns=[event_col])
    return merged


def _merge_comsol_particle_status(comsol: pd.DataFrame, status: pd.DataFrame, state_map: Mapping[str, str]) -> pd.DataFrame:
    if status.empty:
        return comsol
    merged = comsol.merge(status, on="particle_id", how="left")
    status_values = merged.get("comsol_final_status")
    if status_values is not None:
        status_text = status_values.fillna("").astype(str)
        has_status = status_text.str.strip().ne("").to_numpy(dtype=bool)
        mapped = _normalize_comsol_state(status_text.to_numpy(dtype=object), state_map)
        merged["comsol_state"] = np.where(has_status, mapped, merged["comsol_state"].to_numpy(dtype=object))
    return merged


def _counts_frame(left: pd.Series, right: pd.Series, *, left_name: str, right_name: str, key_name: str) -> pd.DataFrame:
    l = left.value_counts(dropna=False).rename(left_name)
    r = right.value_counts(dropna=False).rename(right_name)
    out = pd.concat([l, r], axis=1).fillna(0).astype(int).reset_index()
    out = out.rename(columns={"index": key_name})
    out["delta_solver_minus_comsol"] = out[left_name] - out[right_name]
    return out.sort_values(key_name)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _default_case_file(solver_particles_csv: Path | None, relative_path: str) -> Path | None:
    if solver_particles_csv is None:
        return None
    candidate = solver_particles_csv.parent / relative_path
    return candidate if candidate.exists() else None


def _force_features_from_raw_export(raw_export_dir: Path | None) -> list[dict[str, Any]]:
    if raw_export_dir is None:
        return []
    payload = _read_json_if_exists(raw_export_dir / "physics_feature_inventory.json")
    raw = payload.get("features", [])
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        kind = str(item.get("force_kind", "")).strip()
        if not kind or kind == "other":
            continue
        property_values = item.get("property_values", {})
        if not isinstance(property_values, Mapping):
            property_values = {}
        out.append(
            {
                "component_tag": item.get("component_tag", ""),
                "physics_tag": item.get("physics_tag", ""),
                "physics_label": item.get("physics_label", ""),
                "physics_type": item.get("physics_type", ""),
                "feature_tag": item.get("feature_tag", ""),
                "label": item.get("label", ""),
                "type": item.get("type", ""),
                "force_kind": kind,
                "selection_entities": item.get("selection_entities", []),
                "property_values": dict(property_values),
            }
        )
    return out


def _is_particle_tracing_force_feature(item: Mapping[str, Any]) -> bool:
    physics_tag = str(item.get("physics_tag", "")).strip().lower()
    physics_type = str(item.get("physics_type", "")).strip().lower()
    physics_label = str(item.get("physics_label", "")).strip().lower()
    return (
        physics_tag.startswith(("fpt", "pt"))
        or "particletracing" in physics_type.replace(" ", "")
        or "particle tracing" in physics_label
    )


def _selected_force_fields(raw_export_dir: Path | None) -> dict[str, Any]:
    if raw_export_dir is None:
        return {}
    inventory = _read_json_if_exists(raw_export_dir / "expression_inventory.json")
    selected = inventory.get("selected", {})
    if not isinstance(selected, Mapping):
        return {}
    keys = ("ux", "uy", "uz", "mu", "rho_g", "T", "E_x", "E_y", "E_z", "B_x", "B_y", "B_z")
    return {
        key: value
        for key, value in selected.items()
        if str(key) in keys and isinstance(value, Mapping) and bool(value.get("available", False))
    }


def _solver_force_payloads(solver_output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    diagnostics = _read_json_if_exists(solver_output_dir / "collision_diagnostics.json")
    report = _read_json_if_exists(solver_output_dir / "solver_report.json")
    runtime = diagnostics.get("force_runtime")
    if not isinstance(runtime, Mapping):
        runtime = report.get("force_runtime")
    catalog = diagnostics.get("force_catalog")
    if not isinstance(catalog, Mapping):
        catalog = report.get("force_catalog")
    return (
        dict(runtime) if isinstance(runtime, Mapping) else {},
        dict(catalog) if isinstance(catalog, Mapping) else {},
    )


def _enabled_solver_force_names(force_runtime: Mapping[str, Any], force_catalog: Mapping[str, Any]) -> list[str]:
    names: list[str] = []
    enabled = force_catalog.get("enabled_forces", [])
    if isinstance(enabled, list):
        names.extend(str(value) for value in enabled)
    for key, value in force_runtime.items():
        if not str(key).endswith("_enabled"):
            continue
        try:
            enabled = bool(int(value))
        except (TypeError, ValueError):
            enabled = bool(value)
        if enabled:
            name = str(key).replace("_enabled", "")
            if name == "gravity_buoyancy":
                name = "gravity"
            names.append(name)
    return sorted(set(names))


def _truthy_setting(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on", "enabled"}


def _solver_drag_model(force_catalog: Mapping[str, Any]) -> str:
    models = force_catalog.get("force_models", {})
    if isinstance(models, Mapping):
        model = models.get("drag")
        if model is not None:
            return str(model).strip().lower()
    return "stokes"


def _force_parity_gaps(
    *,
    comsol_features: list[Mapping[str, Any]],
    solver_force_catalog: Mapping[str, Any],
    solver_enabled: list[str],
) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    drag_model = _solver_drag_model(solver_force_catalog)
    enabled = {str(name).strip().lower() for name in solver_enabled}
    force_status = solver_force_catalog.get("force_status", {})
    status_by_force = dict(force_status) if isinstance(force_status, Mapping) else {}
    for feature in comsol_features:
        if str(feature.get("force_kind", "")).strip().lower() != "drag":
            continue
        values = feature.get("property_values", {})
        if not isinstance(values, Mapping):
            values = {}
        rarefaction = str(values.get("Rarefaction_Effects", "")).strip()
        if "cunningham" in rarefaction.lower() and drag_model != "stokes_cunningham":
            gaps.append(
                {
                    "category": "drag_model",
                    "severity": "blocker",
                    "comsol_feature_tag": str(feature.get("feature_tag", "")),
                    "comsol_setting": "Rarefaction_Effects",
                    "comsol_value": rarefaction,
                    "solver_drag_model": drag_model,
                    "recommended_solver_drag_model": "stokes_cunningham",
                    "message": "COMSOL drag enables Cunningham-Millikan-Davies rarefaction but solver drag is not stokes_cunningham.",
                }
            )
        if _truthy_setting(values.get("IncludeVirtualMassAndPressureGradientForces", "")):
            missing = [name for name in ("virtual_mass", "pressure_gradient") if name not in enabled]
            implemented_missing = [
                name for name in missing if str(status_by_force.get(name, "")).strip().lower() == "implemented"
            ]
            unsupported_missing = [name for name in missing if name not in implemented_missing]
            for category, names, message in (
                (
                    "force_not_enabled",
                    implemented_missing,
                    "COMSOL includes virtual mass / pressure-gradient forces; solver counterpart(s) exist but are not enabled",
                ),
                (
                    "unsupported_comsol_contribution",
                    unsupported_missing,
                    "COMSOL includes virtual mass / pressure-gradient forces without a supported solver counterpart",
                ),
            ):
                if not names:
                    continue
                gaps.append(
                    {
                        "category": category,
                        "severity": "blocker",
                        "comsol_feature_tag": str(feature.get("feature_tag", "")),
                        "comsol_setting": "IncludeVirtualMassAndPressureGradientForces",
                        "comsol_value": str(values.get("IncludeVirtualMassAndPressureGradientForces", "")),
                        "solver_counterpart": ",".join(names),
                        "message": f"{message}: {', '.join(names)}.",
                    }
                )
    return gaps


def _force_contribution_alignment_rows(
    *,
    comsol_features: list[Mapping[str, Any]],
    solver_enabled: list[str],
    solver_force_catalog: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    drag_model = _solver_drag_model(solver_force_catalog)
    enabled = {str(name).strip().lower() for name in solver_enabled}
    force_status = solver_force_catalog.get("force_status", {})
    status_by_force = dict(force_status) if isinstance(force_status, Mapping) else {}
    for feature in comsol_features:
        kind = str(feature.get("force_kind", "")).strip().lower()
        if not kind:
            continue
        values = feature.get("property_values", {})
        if not isinstance(values, Mapping):
            values = {}
        status = "supported" if kind in enabled else "missing_solver_force"
        solver_model = drag_model if kind == "drag" else ""
        if kind == "drag":
            rarefaction = str(values.get("Rarefaction_Effects", "")).strip().lower()
            if "cunningham" in rarefaction and drag_model != "stokes_cunningham":
                status = "model_mismatch"
        rows.append(
            {
                "comsol_feature_tag": str(feature.get("feature_tag", "")),
                "comsol_label": str(feature.get("label", "")),
                "force_kind": kind,
                "contribution": kind,
                "solver_status": status,
                "solver_model": solver_model,
                "comsol_setting": "",
                "comsol_value": "",
                "required_solver_inputs": "",
                "notes": "",
            }
        )
        if kind == "drag" and _truthy_setting(values.get("IncludeVirtualMassAndPressureGradientForces", "")):
            for contribution, required_inputs in (
                ("virtual_mass", "fluid_density, particle_density, material_derivative_velocity"),
                ("pressure_gradient", "fluid_velocity, fluid_density, particle_density"),
            ):
                if contribution in enabled:
                    contribution_status = "supported"
                    notes = ""
                elif str(status_by_force.get(contribution, "")).strip().lower() == "implemented":
                    contribution_status = "missing_solver_force"
                    notes = "Implemented by the solver but not enabled in this run configuration."
                else:
                    contribution_status = "unsupported_comsol_contribution"
                    notes = "Recorded as a physics gap; do not tune drag or boundary handling to absorb this contribution."
                rows.append(
                    {
                        "comsol_feature_tag": str(feature.get("feature_tag", "")),
                        "comsol_label": str(feature.get("label", "")),
                        "force_kind": kind,
                        "contribution": contribution,
                        "solver_status": contribution_status,
                        "solver_model": "",
                        "comsol_setting": "IncludeVirtualMassAndPressureGradientForces",
                        "comsol_value": str(values.get("IncludeVirtualMassAndPressureGradientForces", "")),
                        "required_solver_inputs": required_inputs,
                        "notes": notes,
                    }
                )
    return rows


def _write_force_alignment(
    *,
    solver_output_dir: Path,
    raw_export_dir: Path | None,
    out_dir: Path,
) -> dict[str, Any]:
    solver_runtime, solver_catalog = _solver_force_payloads(solver_output_dir)
    solver_enabled = _enabled_solver_force_names(solver_runtime, solver_catalog)
    all_comsol_features = _force_features_from_raw_export(raw_export_dir)
    comsol_features = [item for item in all_comsol_features if _is_particle_tracing_force_feature(item)]
    non_particle_features = [item for item in all_comsol_features if not _is_particle_tracing_force_feature(item)]
    comsol_kinds = sorted({str(item.get("force_kind", "")) for item in comsol_features if item.get("force_kind")})
    non_particle_kinds = sorted({str(item.get("force_kind", "")) for item in non_particle_features if item.get("force_kind")})
    fields = _selected_force_fields(raw_export_dir)
    missing_solver = [name for name in comsol_kinds if name not in solver_enabled]
    physics_gaps = _force_parity_gaps(
        comsol_features=comsol_features,
        solver_force_catalog=solver_catalog,
        solver_enabled=solver_enabled,
    )
    contribution_rows = _force_contribution_alignment_rows(
        comsol_features=comsol_features,
        solver_enabled=solver_enabled,
        solver_force_catalog=solver_catalog,
    )
    missing_export_fields: list[str] = []
    if "thermophoresis" in solver_enabled:
        missing_export_fields.extend([name for name in ("T", "rho_g", "mu") if name not in fields])
    if "dielectrophoresis" in solver_enabled:
        missing_export_fields.extend([name for name in ("E_x", "E_y") if name not in fields])
    if "lift" in solver_enabled:
        missing_export_fields.extend([name for name in ("ux", "uy", "rho_g", "mu") if name not in fields])
    payload = {
        "solver_force_runtime": dict(solver_runtime),
        "solver_force_catalog": dict(solver_catalog),
        "solver_enabled_forces": solver_enabled,
        "comsol_force_kinds": comsol_kinds,
        "comsol_force_features": comsol_features,
        "comsol_non_particle_force_kinds": non_particle_kinds,
        "comsol_non_particle_force_features": non_particle_features,
        "selected_force_fields": fields,
        "comsol_force_without_enabled_solver_counterpart": missing_solver,
        "force_physics_gaps": physics_gaps,
        "force_contribution_alignment": contribution_rows,
        "force_contribution_alignment_csv": str(out_dir / "force_contribution_alignment.csv"),
        "enabled_solver_force_missing_export_field": sorted(set(missing_export_fields)),
    }
    if contribution_rows:
        pd.DataFrame(contribution_rows).to_csv(out_dir / "force_contribution_alignment.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "comsol_feature_tag",
                "comsol_label",
                "force_kind",
                "contribution",
                "solver_status",
                "solver_model",
                "comsol_setting",
                "comsol_value",
                "required_solver_inputs",
                "notes",
            ]
        ).to_csv(out_dir / "force_contribution_alignment.csv", index=False)
    _write_json(out_dir / "force_model_alignment.json", payload)
    return payload


def _release_features_from_raw_export(raw_export_dir: Path | None) -> list[dict[str, Any]]:
    if raw_export_dir is None:
        return []
    payload = _read_json_if_exists(raw_export_dir / "particle_release_inventory.json")
    raw = payload.get("features", [])
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        out.append(
            {
                "component_tag": item.get("component_tag", ""),
                "physics_tag": item.get("physics_tag", ""),
                "feature_tag": item.get("feature_tag", ""),
                "label": item.get("label", ""),
                "type": item.get("type", ""),
                "release_kind": item.get("release_kind", ""),
                "selection_entities": item.get("selection_entities", []),
                "known_settings": item.get("known_settings", {}),
            }
        )
    return out


def _default_solver_particles_csv(solver_output_dir: Path) -> Path | None:
    for name in ("particles.csv", "input_particles.csv", "initial_particles.csv"):
        path = solver_output_dir / name
        if path.exists():
            return path
    return None


def _release_time_values(frame: pd.DataFrame) -> np.ndarray:
    return _numeric_or_nan(frame, ("release_time", "release_time_s", "t_release", "trelease", "time0", "t0", "time"))


def _release_position_frame(frame: pd.DataFrame) -> pd.DataFrame:
    x_col = _first_column(frame, ("x0_m", "x0", "x_m", "x", "r0_m", "r0", "r_m", "r"))
    y_col = _first_column(frame, ("y0_m", "y0", "y_m", "y", "z0_m", "z0", "z_m", "z"))
    z_col = _first_column(frame, ("z0_3d_m", "z0_3d", "z3_m", "z3"))
    data: dict[str, np.ndarray] = {}
    if x_col is not None:
        data["x"] = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=np.float64)
    if y_col is not None:
        data["y"] = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=np.float64)
    if z_col is not None:
        data["z"] = pd.to_numeric(frame[z_col], errors="coerce").to_numpy(dtype=np.float64)
    return pd.DataFrame(data)


def _release_velocity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "v_x": ("v_x0", "vx0", "v_x", "vx", "u0"),
        "v_y": ("v_y0", "vy0", "v_y", "vy", "v0", "w0"),
        "v_z": ("v_z0", "vz0", "v_z", "vz"),
    }
    data: dict[str, np.ndarray] = {}
    for key, names in aliases.items():
        values = _numeric_or_nan(frame, names)
        if np.isfinite(values).any():
            data[key] = values
    return pd.DataFrame(data)


def _release_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"available": False}
    frame = _read_csv(path)
    times = _release_time_values(frame)
    finite_times = times[np.isfinite(times)]
    pos = _release_position_frame(frame)
    vel = _release_velocity_frame(frame)
    source_part = _numeric_or_nan(frame, ("source_part_id", "part_id", "boundary_id", "source_boundary_id"))
    source_finite = source_part[np.isfinite(source_part)].astype(int)
    source_counts = {str(int(v)): int(np.count_nonzero(source_finite == v)) for v in sorted(set(source_finite.tolist()))}
    return {
        "available": True,
        "path": str(path),
        "particle_count": int(len(frame)),
        "release_time_s": _finite_summary(times),
        "unique_release_time_count": int(np.unique(finite_times).size) if finite_times.size else 0,
        "position_bounds": _bounds_summary(pos),
        "velocity_bounds": _bounds_summary(vel),
        "source_part_counts": source_counts,
    }


def _release_error_summary(solver_particles_csv: Path | None, comsol_release_csv: Path | None) -> dict[str, Any]:
    if solver_particles_csv is None or comsol_release_csv is None or not solver_particles_csv.exists() or not comsol_release_csv.exists():
        return {"available": False}
    solver = _read_csv(solver_particles_csv)
    comsol = _read_csv(comsol_release_csv)
    try:
        solver_id = _particle_id(solver)
        comsol_id = _particle_id(comsol)
    except ValueError:
        return {"available": False, "reason": "particle_id missing"}
    s = solver.copy()
    c = comsol.copy()
    s["_particle_id"] = solver_id
    c["_particle_id"] = comsol_id
    s_prefixed = s.rename(columns={col: f"{col}_solver" for col in s.columns if col != "_particle_id"})
    c_prefixed = c.rename(columns={col: f"{col}_comsol" for col in c.columns if col != "_particle_id"})
    matched = s_prefixed.merge(c_prefixed, on="_particle_id", how="inner")
    if matched.empty:
        return {"available": False, "reason": "no matched particle_id"}
    solver_release = matched[[c for c in matched.columns if c.endswith("_solver")]].rename(
        columns=lambda c: c.removesuffix("_solver")
    )
    comsol_release = matched[[c for c in matched.columns if c.endswith("_comsol")]].rename(
        columns=lambda c: c.removesuffix("_comsol")
    )
    solver_time = _release_time_values(solver_release)
    comsol_time = _release_time_values(comsol_release)
    solver_pos = _release_position_frame(solver_release)
    comsol_pos = _release_position_frame(comsol_release)
    solver_vel = _release_velocity_frame(solver_release)
    comsol_vel = _release_velocity_frame(comsol_release)
    solver_source = _numeric_or_nan(solver_release, ("source_part_id", "part_id", "boundary_id", "source_boundary_id", "source_entity"))
    comsol_source = _numeric_or_nan(comsol_release, ("source_part_id", "part_id", "boundary_id", "source_boundary_id", "source_entity"))
    source_valid = np.isfinite(solver_source) & np.isfinite(comsol_source)
    return {
        "available": True,
        "matched_particle_count": int(len(matched)),
        "release_time_error_s": _finite_summary(np.abs(solver_time - comsol_time)),
        "release_position_error_m": _finite_summary(_norm_error(solver_pos, comsol_pos)),
        "release_velocity_error_mps": _finite_summary(_norm_error(solver_vel, comsol_vel)),
        "source_entity_match_ratio": (
            float(np.mean(solver_source[source_valid].astype(int) == comsol_source[source_valid].astype(int)))
            if np.any(source_valid)
            else None
        ),
    }


def _write_release_alignment(
    *,
    solver_output_dir: Path,
    raw_export_dir: Path | None,
    out_dir: Path,
    solver_particles_csv: Path | None,
    comsol_release_csv: Path | None,
) -> dict[str, Any]:
    solver_particles = solver_particles_csv if solver_particles_csv is not None else _default_solver_particles_csv(solver_output_dir)
    features = _release_features_from_raw_export(raw_export_dir)
    payload = {
        "comsol_release_features": features,
        "comsol_release_feature_count": int(len(features)),
        "comsol_release_kinds": sorted({str(item.get("release_kind", "")) for item in features if item.get("release_kind")}),
        "solver_particles": _release_summary(solver_particles),
        "comsol_release_particles": _release_summary(comsol_release_csv),
        "matched_release_errors": _release_error_summary(solver_particles, comsol_release_csv),
    }
    _write_json(out_dir / "release_alignment.json", payload)
    return payload


def _solver_positions_path(output_dir: Path) -> Path | None:
    for name in ("positions_2d.npy", "positions_3d.npy"):
        path = output_dir / name
        if path.exists():
            return path
    return None


def _time_values(frame: pd.DataFrame) -> np.ndarray:
    return _numeric_or_nan(frame, ("time_s", "time", "t"))


def _load_solver_trajectory(output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    positions_path = _solver_positions_path(output_dir)
    frames_path = output_dir / "save_frames.csv"
    final_path = output_dir / "final_particles.csv"
    if positions_path is None or not frames_path.exists() or not final_path.exists():
        return None
    positions = np.load(positions_path)
    frames = _read_csv(frames_path)
    times = _time_values(frames)
    particle_ids = _particle_id(_read_csv(final_path))
    if positions.shape[0] != times.size:
        raise ValueError(f"solver trajectory frame count mismatch: positions={positions.shape[0]}, save_frames={times.size}")
    if positions.shape[1] != particle_ids.size:
        raise ValueError(f"solver trajectory particle count mismatch: positions={positions.shape[1]}, particles={particle_ids.size}")
    return np.asarray(positions, dtype=np.float64), np.asarray(times, dtype=np.float64), np.asarray(particle_ids, dtype=np.int64)


def _nearest_index(values: np.ndarray, target: float) -> int:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return -1
    return int(np.nanargmin(np.abs(arr - float(target))))


def _write_trajectory_alignment(
    *,
    solver_output_dir: Path,
    comsol_trajectory_csv: Path | None,
    out_dir: Path,
) -> dict[str, Any]:
    if comsol_trajectory_csv is None or not comsol_trajectory_csv.exists():
        payload = {"available": False, "reason": "comsol trajectory CSV not provided"}
        _write_json(out_dir / "trajectory_alignment.json", payload)
        return payload
    solver_loaded = _load_solver_trajectory(solver_output_dir)
    if solver_loaded is None:
        payload = {"available": False, "reason": "solver positions/save_frames/final_particles are incomplete"}
        _write_json(out_dir / "trajectory_alignment.json", payload)
        return payload
    positions, solver_times, solver_particle_ids = solver_loaded
    particle_index = {int(pid): i for i, pid in enumerate(solver_particle_ids.tolist())}
    raw = _read_csv(comsol_trajectory_csv)
    raw = raw.copy()
    raw["_particle_id"] = _particle_id(raw)
    raw["_time_s"] = _time_values(raw)
    comsol_pos = _position_frame(raw)
    rows: list[dict[str, Any]] = []
    for row_index, row in raw.iterrows():
        pid = int(row["_particle_id"])
        t = float(row["_time_s"])
        pidx = particle_index.get(pid)
        if pidx is None or not np.isfinite(t):
            continue
        tidx = _nearest_index(solver_times, t)
        if tidx < 0:
            continue
        solver_pos = positions[tidx, pidx, : comsol_pos.shape[1]]
        cpos = comsol_pos.iloc[row_index].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(cpos)) or not np.all(np.isfinite(solver_pos)):
            continue
        rows.append(
            {
                "particle_id": pid,
                "comsol_time_s": t,
                "solver_time_s": float(solver_times[tidx]),
                "time_error_s": abs(float(solver_times[tidx]) - t),
                "position_error_m": float(np.linalg.norm(solver_pos - cpos)),
            }
        )
    error_frame = pd.DataFrame(rows)
    error_frame.to_csv(out_dir / "matched_trajectory_errors.csv", index=False)
    distribution = _distribution_alignment(raw, positions, solver_times, particle_index, comsol_pos)
    distribution.to_csv(out_dir / "distribution_alignment.csv", index=False)
    payload = {
        "available": True,
        "comsol_trajectory_csv": str(comsol_trajectory_csv),
        "matched_sample_count": int(len(error_frame)),
        "matched_particle_count": int(error_frame["particle_id"].nunique()) if not error_frame.empty else 0,
        "time_error_s": _finite_summary(error_frame["time_error_s"].to_numpy(dtype=np.float64)) if not error_frame.empty else {"count": 0},
        "position_error_m": _finite_summary(error_frame["position_error_m"].to_numpy(dtype=np.float64)) if not error_frame.empty else {"count": 0},
        "distribution_time_count": int(len(distribution)),
    }
    _write_json(out_dir / "trajectory_alignment.json", payload)
    return payload


def _distribution_alignment(
    raw: pd.DataFrame,
    solver_positions: np.ndarray,
    solver_times: np.ndarray,
    particle_index: Mapping[int, int],
    comsol_pos: pd.DataFrame,
) -> pd.DataFrame:
    if raw.empty or comsol_pos.empty:
        return pd.DataFrame(columns=["comsol_time_s"])
    work = raw[["_particle_id", "_time_s"]].copy()
    for col in comsol_pos.columns:
        work[f"comsol_{col}"] = comsol_pos[col].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for t, sub in work.groupby("_time_s", dropna=True):
        if not np.isfinite(float(t)):
            continue
        tidx = _nearest_index(solver_times, float(t))
        solver_points = []
        comsol_points = []
        for _, row in sub.iterrows():
            pidx = particle_index.get(int(row["_particle_id"]))
            if pidx is None:
                continue
            c = np.asarray([row[f"comsol_{col}"] for col in comsol_pos.columns], dtype=np.float64)
            s = solver_positions[tidx, pidx, : c.size]
            if np.all(np.isfinite(c)) and np.all(np.isfinite(s)):
                comsol_points.append(c)
                solver_points.append(s)
        if not solver_points:
            rows.append({"comsol_time_s": float(t), "matched_count": 0})
            continue
        c_arr = np.vstack(comsol_points)
        s_arr = np.vstack(solver_points)
        c_centroid = np.mean(c_arr, axis=0)
        s_centroid = np.mean(s_arr, axis=0)
        rows.append(
            {
                "comsol_time_s": float(t),
                "solver_time_s": float(solver_times[tidx]),
                "matched_count": int(c_arr.shape[0]),
                "centroid_error_m": float(np.linalg.norm(s_centroid - c_centroid)),
                "rms_position_error_m": float(np.sqrt(np.mean(np.sum((s_arr - c_arr) ** 2, axis=1)))),
            }
        )
    return pd.DataFrame(rows)


def _field_quantity_at_times(values: np.ndarray, times: np.ndarray, sample_times: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    field_times = np.asarray(times, dtype=np.float64)
    if arr.ndim == 2:
        return arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"field quantity must be 2D or time+2D, got shape {arr.shape}")
    if field_times.size <= 1 or arr.shape[0] == 1:
        return arr[:1, :, :]
    indices = np.searchsorted(field_times, np.asarray(sample_times, dtype=np.float64), side="right") - 1
    indices = np.clip(indices, 0, field_times.size - 1)
    return arr[indices, :, :]


def _bilinear_regular_2d(
    *,
    axis_0: np.ndarray,
    axis_1: np.ndarray,
    values: np.ndarray,
    valid_mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ax0 = np.asarray(axis_0, dtype=np.float64)
    ax1 = np.asarray(axis_1, dtype=np.float64)
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    ix = np.searchsorted(ax0, xs, side="right") - 1
    iy = np.searchsorted(ax1, ys, side="right") - 1
    inside = (ix >= 0) & (iy >= 0) & (ix < ax0.size - 1) & (iy < ax1.size - 1)
    out = np.full(xs.shape, np.nan, dtype=np.float64)
    clean = np.zeros(xs.shape, dtype=bool)
    if not np.any(inside):
        return out, inside, clean
    rows = np.flatnonzero(inside)
    i = ix[rows]
    j = iy[rows]
    stencil_valid = (
        valid_mask[i, j]
        & valid_mask[i + 1, j]
        & valid_mask[i, j + 1]
        & valid_mask[i + 1, j + 1]
    )
    finite_values = (
        np.isfinite(values[i, j])
        & np.isfinite(values[i + 1, j])
        & np.isfinite(values[i, j + 1])
        & np.isfinite(values[i + 1, j + 1])
    )
    ok_local = stencil_valid & finite_values
    if np.any(ok_local):
        ok_rows = rows[ok_local]
        ii = ix[ok_rows]
        jj = iy[ok_rows]
        tx = (xs[ok_rows] - ax0[ii]) / (ax0[ii + 1] - ax0[ii])
        ty = (ys[ok_rows] - ax1[jj]) / (ax1[jj + 1] - ax1[jj])
        out[ok_rows] = (
            (1.0 - tx) * (1.0 - ty) * values[ii, jj]
            + tx * (1.0 - ty) * values[ii + 1, jj]
            + (1.0 - tx) * ty * values[ii, jj + 1]
            + tx * ty * values[ii + 1, jj + 1]
        )
        clean[ok_rows] = True
    return out, inside, clean


def _field_alignment_outputs(
    *,
    work: pd.DataFrame,
    residual: np.ndarray,
    out_dir: Path,
    payload_base: Mapping[str, Any],
    inside_name: str = "inside_grid",
    clean_name: str = "clean_stencil",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_rows = []
    for source_part_id, sub in work.groupby("source_part_id", dropna=False):
        stats = _finite_summary(sub["velocity_residual_mps"].to_numpy(dtype=np.float64))
        source_rows.append(
            {
                "source_part_id": source_part_id,
                "sample_count": int(len(sub)),
                "inside_grid_fraction": float(np.mean(sub[inside_name].to_numpy(dtype=float))) if len(sub) else 0.0,
                "clean_stencil_fraction": float(np.mean(sub[clean_name].to_numpy(dtype=float))) if len(sub) else 0.0,
                "support_fraction": float(np.mean(sub[clean_name].to_numpy(dtype=float))) if len(sub) else 0.0,
                "velocity_residual_mean_mps": stats.get("mean", np.nan),
                "velocity_residual_median_mps": stats.get("median", np.nan),
                "velocity_residual_p90_mps": stats.get("p90", np.nan),
                "velocity_residual_max_mps": stats.get("max", np.nan),
            }
        )
    by_source = pd.DataFrame(source_rows)
    by_time = (
        work.groupby("time_s", dropna=True)
        .agg(
            sample_count=("particle_id", "count"),
            inside_grid_fraction=(inside_name, "mean"),
            clean_stencil_fraction=(clean_name, "mean"),
            support_fraction=(clean_name, "mean"),
            velocity_residual_mean_mps=("velocity_residual_mps", "mean"),
            velocity_residual_median_mps=("velocity_residual_mps", "median"),
            velocity_residual_max_mps=("velocity_residual_mps", "max"),
        )
        .reset_index()
    )
    by_source.to_csv(out_dir / "field_alignment_by_source.csv", index=False)
    by_time.to_csv(out_dir / "field_alignment_by_time.csv", index=False)
    clean_values = work[clean_name].to_numpy(dtype=float)
    inside_values = work[inside_name].to_numpy(dtype=float)
    payload = {
        **dict(payload_base),
        "sample_count": int(len(work)),
        "inside_grid_fraction": float(np.mean(inside_values)) if len(work) else 0.0,
        "clean_stencil_fraction": float(np.mean(clean_values)) if len(work) else 0.0,
        "support_fraction": float(np.mean(clean_values)) if len(work) else 0.0,
        "velocity_residual_mps": _finite_summary(residual),
        "source_count": int(by_source["source_part_id"].nunique(dropna=True)) if not by_source.empty else 0,
        "outputs": {
            "field_alignment_by_source_csv": str(out_dir / "field_alignment_by_source.csv"),
            "field_alignment_by_time_csv": str(out_dir / "field_alignment_by_time.csv"),
        },
    }
    _write_json(out_dir / "field_alignment.json", payload)
    return payload


def _write_triangle_mesh_field_alignment(
    *,
    field_npz: Path,
    comsol_trajectory_csv: Path,
    solver_particles_csv: Path | None,
    out_dir: Path,
) -> dict[str, Any]:
    try:
        provider = build_precomputed_triangle_mesh_field(
            {"npz_path": str(field_npz)},
            spatial_dim=2,
            coordinate_system="cartesian_xy",
            gas_density_kgm3=1.0,
        )
    except Exception as exc:
        payload = {"available": False, "reason": f"triangle mesh field load failed: {exc}"}
        _write_json(out_dir / "field_alignment.json", payload)
        return payload
    field = provider.field
    if "ux" not in field.quantities or "uy" not in field.quantities:
        payload = {"available": False, "reason": "triangle mesh field npz must contain ux and uy"}
        _write_json(out_dir / "field_alignment.json", payload)
        return payload
    field_times = np.asarray(getattr(field.quantities["ux"], "times", np.asarray([0.0])), dtype=np.float64)

    raw = _read_csv(comsol_trajectory_csv)
    raw = raw.copy()
    raw["_particle_id"] = _particle_id(raw)
    raw["_time_s"] = _time_values(raw)
    pos = _position_frame(raw)
    vel = _velocity_frame(raw)
    if not {"x", "y"}.issubset(set(pos.columns)) or not {"v_x", "v_y"}.issubset(set(vel.columns)):
        payload = {"available": False, "reason": "trajectory must contain x/y and v_x/v_y columns"}
        _write_json(out_dir / "field_alignment.json", payload)
        return payload

    sample_times = raw["_time_s"].to_numpy(dtype=np.float64)
    positions = pos[["x", "y"]].to_numpy(dtype=np.float64)
    ux_sample = np.full(len(raw), np.nan, dtype=np.float64)
    uy_sample = np.full(len(raw), np.nan, dtype=np.float64)
    clean = np.zeros(len(raw), dtype=bool)
    for i in range(len(raw)):
        status = int(sample_triangle_mesh_status(field, positions[i]))
        clean[i] = status == int(VALID_MASK_STATUS_CLEAN)
        if clean[i]:
            ux_sample[i] = float(sample_triangle_mesh_series(field.quantities["ux"], field, positions[i], float(sample_times[i])))
            uy_sample[i] = float(sample_triangle_mesh_series(field.quantities["uy"], field, positions[i], float(sample_times[i])))

    residual = np.sqrt(
        (ux_sample - vel["v_x"].to_numpy(dtype=np.float64)) ** 2
        + (uy_sample - vel["v_y"].to_numpy(dtype=np.float64)) ** 2
    )
    work = pd.DataFrame(
        {
            "particle_id": raw["_particle_id"].to_numpy(dtype=np.int64),
            "time_s": sample_times,
            "inside_grid": clean.astype(int),
            "clean_stencil": clean.astype(int),
            "velocity_residual_mps": residual,
        }
    )
    if solver_particles_csv is not None and solver_particles_csv.exists():
        particles = _read_csv(solver_particles_csv)
        particles["particle_id"] = _particle_id(particles)
        source_col = _first_column(particles, ("source_part_id", "source_entity", "part_id"))
        if source_col is not None:
            source = particles[["particle_id", source_col]].rename(columns={source_col: "source_part_id"})
            work = work.merge(source, on="particle_id", how="left")
    if "source_part_id" not in work.columns:
        work["source_part_id"] = np.nan

    return _field_alignment_outputs(
        work=work,
        residual=residual,
        out_dir=out_dir,
        payload_base={
            "available": True,
            "field_backend_kind": "triangle_mesh_2d",
            "field_npz": str(field_npz),
            "comsol_trajectory_csv": str(comsol_trajectory_csv),
            "mesh_vertex_count": int(field.mesh_vertices.shape[0]),
            "mesh_triangle_count": int(field.mesh_triangles.shape[0]),
            "field_time_count": int(field_times.size),
            "field_time_min_s": float(np.nanmin(field_times)) if field_times.size else None,
            "field_time_max_s": float(np.nanmax(field_times)) if field_times.size else None,
        },
    )


def _write_field_alignment(
    *,
    field_npz: Path | None,
    comsol_trajectory_csv: Path | None,
    solver_particles_csv: Path | None,
    out_dir: Path,
) -> dict[str, Any]:
    if field_npz is None or not field_npz.exists():
        payload = {"available": False, "reason": "field npz not provided"}
        _write_json(out_dir / "field_alignment.json", payload)
        return payload
    if comsol_trajectory_csv is None or not comsol_trajectory_csv.exists():
        payload = {"available": False, "reason": "COMSOL trajectory CSV not provided"}
        _write_json(out_dir / "field_alignment.json", payload)
        return payload
    with np.load(field_npz) as payload_npz:
        keys = set(payload_npz.files)
        if {"mesh_vertices", "mesh_triangles"}.issubset(keys):
            return _write_triangle_mesh_field_alignment(
                field_npz=field_npz,
                comsol_trajectory_csv=comsol_trajectory_csv,
                solver_particles_csv=solver_particles_csv,
                out_dir=out_dir,
            )
        axis_0 = np.asarray(payload_npz["axis_0"], dtype=np.float64)
        axis_1 = np.asarray(payload_npz["axis_1"], dtype=np.float64)
        valid_mask = np.asarray(payload_npz["valid_mask"], dtype=bool)
        times = np.asarray(payload_npz["times"], dtype=np.float64) if "times" in payload_npz else np.asarray([0.0])
        ux_all = np.asarray(payload_npz["ux"], dtype=np.float64)
        uy_all = np.asarray(payload_npz["uy"], dtype=np.float64)
    raw = _read_csv(comsol_trajectory_csv)
    raw = raw.copy()
    raw["_particle_id"] = _particle_id(raw)
    raw["_time_s"] = _time_values(raw)
    pos = _position_frame(raw)
    vel = _velocity_frame(raw)
    if not {"x", "y"}.issubset(set(pos.columns)) or not {"v_x", "v_y"}.issubset(set(vel.columns)):
        payload = {"available": False, "reason": "trajectory must contain x/y and v_x/v_y columns"}
        _write_json(out_dir / "field_alignment.json", payload)
        return payload

    sample_times = raw["_time_s"].to_numpy(dtype=np.float64)
    ux_time = _field_quantity_at_times(ux_all, times, sample_times)
    uy_time = _field_quantity_at_times(uy_all, times, sample_times)
    x = pos["x"].to_numpy(dtype=np.float64)
    y = pos["y"].to_numpy(dtype=np.float64)
    ux_sample = np.full(len(raw), np.nan, dtype=np.float64)
    uy_sample = np.full(len(raw), np.nan, dtype=np.float64)
    inside = np.zeros(len(raw), dtype=bool)
    clean = np.zeros(len(raw), dtype=bool)
    if ux_time.shape[0] == 1:
        ux_sample, inside, clean_x = _bilinear_regular_2d(
            axis_0=axis_0,
            axis_1=axis_1,
            values=ux_time[0],
            valid_mask=valid_mask,
            x=x,
            y=y,
        )
        uy_sample, inside_y, clean_y = _bilinear_regular_2d(
            axis_0=axis_0,
            axis_1=axis_1,
            values=uy_time[0],
            valid_mask=valid_mask,
            x=x,
            y=y,
        )
        inside = inside & inside_y
        clean = clean_x & clean_y
    else:
        for time_index in np.unique(np.arange(ux_time.shape[0])):
            rows = np.flatnonzero(np.arange(ux_time.shape[0]) == int(time_index))
            if rows.size == 0:
                continue
            ux_rows, inside_rows, clean_x = _bilinear_regular_2d(
                axis_0=axis_0,
                axis_1=axis_1,
                values=ux_time[int(time_index)],
                valid_mask=valid_mask,
                x=x[rows],
                y=y[rows],
            )
            uy_rows, inside_y, clean_y = _bilinear_regular_2d(
                axis_0=axis_0,
                axis_1=axis_1,
                values=uy_time[int(time_index)],
                valid_mask=valid_mask,
                x=x[rows],
                y=y[rows],
            )
            ux_sample[rows] = ux_rows
            uy_sample[rows] = uy_rows
            inside[rows] = inside_rows & inside_y
            clean[rows] = clean_x & clean_y

    residual = np.sqrt((ux_sample - vel["v_x"].to_numpy(dtype=np.float64)) ** 2 + (uy_sample - vel["v_y"].to_numpy(dtype=np.float64)) ** 2)
    work = pd.DataFrame(
        {
            "particle_id": raw["_particle_id"].to_numpy(dtype=np.int64),
            "time_s": sample_times,
            "inside_grid": inside.astype(int),
            "clean_stencil": clean.astype(int),
            "velocity_residual_mps": residual,
        }
    )
    if solver_particles_csv is not None and solver_particles_csv.exists():
        particles = _read_csv(solver_particles_csv)
        particles["particle_id"] = _particle_id(particles)
        source_col = _first_column(particles, ("source_part_id", "source_entity", "part_id"))
        if source_col is not None:
            source = particles[["particle_id", source_col]].rename(columns={source_col: "source_part_id"})
            work = work.merge(source, on="particle_id", how="left")
    if "source_part_id" not in work.columns:
        work["source_part_id"] = np.nan

    return _field_alignment_outputs(
        work=work,
        residual=residual,
        out_dir=out_dir,
        payload_base={
            "available": True,
            "field_backend_kind": "regular_rectilinear",
            "field_npz": str(field_npz),
            "comsol_trajectory_csv": str(comsol_trajectory_csv),
            "field_time_count": int(times.size),
            "field_time_min_s": float(np.nanmin(times)) if times.size else None,
            "field_time_max_s": float(np.nanmax(times)) if times.size else None,
        },
    )


def write_field_alignment(
    *,
    field_npz: str | Path | None,
    comsol_trajectory_csv: str | Path | None,
    solver_particles_csv: str | Path | None,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Replay an exported solver field on a canonical COMSOL trajectory table."""

    return _write_field_alignment(
        field_npz=Path(field_npz) if field_npz is not None else None,
        comsol_trajectory_csv=Path(comsol_trajectory_csv) if comsol_trajectory_csv is not None else None,
        solver_particles_csv=Path(solver_particles_csv) if solver_particles_csv is not None else None,
        out_dir=Path(out_dir),
    )


def _write_trend_alignment(
    *,
    solver_output_dir: Path,
    comsol_trajectory_csv: Path | None,
    solver_particles_csv: Path | None,
    out_dir: Path,
) -> dict[str, Any]:
    if comsol_trajectory_csv is None or not comsol_trajectory_csv.exists():
        payload = {"available": False, "reason": "COMSOL trajectory CSV not provided"}
        _write_json(out_dir / "trend_alignment.json", payload)
        return payload
    final_path = solver_output_dir / "final_particles.csv"
    if not final_path.exists():
        payload = {"available": False, "reason": "solver final_particles.csv missing"}
        _write_json(out_dir / "trend_alignment.json", payload)
        return payload
    raw = _read_csv(comsol_trajectory_csv)
    raw = raw.copy()
    raw["particle_id"] = _particle_id(raw)
    raw["time_s"] = _time_values(raw)
    final_time = float(np.nanmax(raw["time_s"].to_numpy(dtype=np.float64))) if len(raw) else float("nan")
    last = raw.groupby("particle_id", as_index=False)["time_s"].max().rename(columns={"time_s": "comsol_last_time_s"})
    final = _read_csv(final_path)
    final["particle_id"] = _particle_id(final)
    final_labels = pd.DataFrame({"particle_id": final["particle_id"], "solver_state": _solver_state_labels(final)})
    trend = last.merge(final_labels, on="particle_id", how="outer")
    if solver_particles_csv is not None and solver_particles_csv.exists():
        particles = _read_csv(solver_particles_csv)
        particles["particle_id"] = _particle_id(particles)
        source_col = _first_column(particles, ("source_part_id", "source_entity", "part_id"))
        release_col = _first_column(particles, ("release_time", "release_time_s", "t0", "time_s"))
        cols = ["particle_id"]
        rename = {}
        if source_col is not None:
            cols.append(source_col)
            rename[source_col] = "source_part_id"
        if release_col is not None:
            cols.append(release_col)
            rename[release_col] = "release_time_s"
        trend = trend.merge(particles[cols].rename(columns=rename), on="particle_id", how="left")
    if "source_part_id" not in trend.columns:
        trend["source_part_id"] = np.nan
    trend["comsol_finite_at_final_time"] = (
        np.isfinite(trend["comsol_last_time_s"].to_numpy(dtype=np.float64))
        & (np.abs(trend["comsol_last_time_s"].to_numpy(dtype=np.float64) - final_time) <= 1.0e-12)
    ).astype(int)
    rows = []
    for source_part_id, sub in trend.groupby("source_part_id", dropna=False):
        last_times = sub["comsol_last_time_s"].to_numpy(dtype=np.float64)
        state_counts = sub["solver_state"].value_counts(dropna=False).to_dict()
        row = {
            "source_part_id": source_part_id,
            "particle_count": int(len(sub)),
            "comsol_finite_at_final_fraction": float(np.mean(sub["comsol_finite_at_final_time"].to_numpy(dtype=float))) if len(sub) else 0.0,
            "comsol_last_time_mean_s": float(np.nanmean(last_times)) if np.isfinite(last_times).any() else np.nan,
            "comsol_last_time_median_s": float(np.nanmedian(last_times)) if np.isfinite(last_times).any() else np.nan,
        }
        for state, count in sorted(state_counts.items()):
            row[f"solver_state_{state}_count"] = int(count)
        rows.append(row)
    by_source = pd.DataFrame(rows)
    by_source.to_csv(out_dir / "trend_alignment_by_source.csv", index=False)
    payload = {
        "available": True,
        "comsol_trajectory_csv": str(comsol_trajectory_csv),
        "solver_final_particles_csv": str(final_path),
        "particle_count": int(len(trend)),
        "comsol_output_final_time_s": final_time,
        "comsol_finite_at_final_count": int(trend["comsol_finite_at_final_time"].sum()),
        "comsol_last_time_s": _finite_summary(trend["comsol_last_time_s"].to_numpy(dtype=np.float64)),
        "solver_state_counts": {str(k): int(v) for k, v in trend["solver_state"].value_counts(dropna=False).sort_index().items()},
        "source_count": int(by_source["source_part_id"].nunique(dropna=True)) if not by_source.empty else 0,
        "outputs": {"trend_alignment_by_source_csv": str(out_dir / "trend_alignment_by_source.csv")},
    }
    _write_json(out_dir / "trend_alignment.json", payload)
    return payload


def _first_wall_events(solver_output_dir: Path) -> pd.DataFrame:
    path = solver_output_dir / "wall_events.csv"
    columns = [
        "particle_id",
        "first_wall_time_s",
        "first_wall_part_id",
        "first_wall_outcome",
        "wall_event_count",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    wall = _read_csv(path)
    if wall.empty or "particle_id" not in wall.columns:
        return pd.DataFrame(columns=columns)
    time_col = _first_column(wall, ("hit_time_s", "time_s"))
    if time_col is None:
        return pd.DataFrame(columns=columns)
    work = wall.copy()
    work["particle_id"] = _particle_id(work)
    work["_wall_time_s"] = pd.to_numeric(work[time_col], errors="coerce")
    work = work[np.isfinite(work["_wall_time_s"].to_numpy(dtype=np.float64))]
    if work.empty:
        return pd.DataFrame(columns=columns)
    work = work.sort_values(["particle_id", "_wall_time_s"], kind="mergesort")
    first = work.groupby("particle_id", as_index=False).first()
    counts = work.groupby("particle_id", as_index=False).size().rename(columns={"size": "wall_event_count"})
    part_col = _first_column(first, ("part_id", "solver_hit_part_id", "boundary_part_id"))
    outcome_col = _first_column(first, ("outcome", "wall_outcome"))
    out = pd.DataFrame(
        {
            "particle_id": first["particle_id"].to_numpy(dtype=np.int64),
            "first_wall_time_s": first["_wall_time_s"].to_numpy(dtype=np.float64),
            "first_wall_part_id": pd.to_numeric(first[part_col], errors="coerce").to_numpy(dtype=np.float64)
            if part_col is not None
            else np.nan,
            "first_wall_outcome": first[outcome_col].astype(str).to_numpy() if outcome_col is not None else "",
        }
    )
    return out.merge(counts, on="particle_id", how="left")


def _write_divergence_alignment(
    *,
    solver_output_dir: Path,
    solver_particles_csv: Path | None,
    out_dir: Path,
    thresholds_m: tuple[float, ...] = (1.0e-4, 5.0e-4, 1.0e-3),
    coincidence_window_s: float = 5.0e-3,
) -> dict[str, Any]:
    errors_path = out_dir / "matched_trajectory_errors.csv"
    if not errors_path.exists():
        payload = {"available": False, "reason": "matched trajectory errors missing"}
        _write_json(out_dir / "divergence_alignment.json", payload)
        return payload
    errors = _read_csv(errors_path)
    if errors.empty or "particle_id" not in errors.columns or "position_error_m" not in errors.columns:
        payload = {"available": False, "reason": "trajectory error table missing required columns"}
        _write_json(out_dir / "divergence_alignment.json", payload)
        return payload
    time_col = _first_column(errors, ("comsol_time_s", "time_s", "solver_time_s"))
    if time_col is None:
        payload = {"available": False, "reason": "trajectory error table missing time column"}
        _write_json(out_dir / "divergence_alignment.json", payload)
        return payload

    work = errors[["particle_id", time_col, "position_error_m"]].copy()
    work["particle_id"] = _particle_id(work)
    work["time_s"] = pd.to_numeric(work[time_col], errors="coerce")
    work["position_error_m"] = pd.to_numeric(work["position_error_m"], errors="coerce")
    particle_ids = pd.DataFrame({"particle_id": np.sort(work["particle_id"].dropna().astype(int).unique())})

    final_path = solver_output_dir / "final_particles.csv"
    if final_path.exists():
        final = _read_csv(final_path)
        final["particle_id"] = _particle_id(final)
        labels = pd.DataFrame({"particle_id": final["particle_id"], "solver_state": _solver_state_labels(final)})
        particle_ids = particle_ids.merge(labels, on="particle_id", how="left")
    else:
        particle_ids["solver_state"] = ""

    if solver_particles_csv is not None and solver_particles_csv.exists():
        particles = _read_csv(solver_particles_csv)
        particles["particle_id"] = _particle_id(particles)
        source_col = _first_column(particles, ("source_part_id", "source_entity", "part_id"))
        if source_col is not None:
            particle_ids = particle_ids.merge(
                particles[["particle_id", source_col]].rename(columns={source_col: "source_part_id"}),
                on="particle_id",
                how="left",
            )
    if "source_part_id" not in particle_ids.columns:
        particle_ids["source_part_id"] = np.nan

    wall_first = _first_wall_events(solver_output_dir)
    particle_base = particle_ids.merge(wall_first, on="particle_id", how="left")
    row_frames: list[pd.DataFrame] = []
    summary_by_threshold: dict[str, Any] = {}
    for threshold in thresholds_m:
        exceeded = work[work["position_error_m"] >= float(threshold)]
        first_div = (
            exceeded.sort_values(["particle_id", "time_s"], kind="mergesort")
            .groupby("particle_id", as_index=False)
            .first()[["particle_id", "time_s", "position_error_m"]]
            .rename(
                columns={
                    "time_s": "first_divergence_time_s",
                    "position_error_m": "first_divergence_error_m",
                }
            )
        )
        rows = particle_base.merge(first_div, on="particle_id", how="left")
        div_time = rows["first_divergence_time_s"].to_numpy(dtype=np.float64)
        wall_time = rows["first_wall_time_s"].to_numpy(dtype=np.float64)
        has_div = np.isfinite(div_time)
        has_wall = np.isfinite(wall_time)
        cls = np.full(len(rows), "no_divergence", dtype=object)
        cls[has_div & ~has_wall] = "diverged_without_solver_wall_event"
        cls[has_div & has_wall & (div_time < wall_time - float(coincidence_window_s))] = "diverged_before_first_wall"
        near = has_div & has_wall & (np.abs(div_time - wall_time) <= float(coincidence_window_s))
        cls[near] = "diverged_near_first_wall"
        cls[has_div & has_wall & (div_time > wall_time + float(coincidence_window_s))] = "diverged_after_first_wall"
        rows["threshold_m"] = float(threshold)
        rows["divergence_wall_relation"] = cls
        row_frames.append(rows)
        counts = rows["divergence_wall_relation"].value_counts(dropna=False).sort_index().to_dict()
        state_counts = (
            rows.loc[has_div, "solver_state"].value_counts(dropna=False).sort_index().to_dict()
            if "solver_state" in rows.columns
            else {}
        )
        source_counts = (
            rows.loc[has_div, "source_part_id"].value_counts(dropna=False).sort_index().to_dict()
            if "source_part_id" in rows.columns
            else {}
        )
        summary_by_threshold[str(threshold)] = {
            "threshold_m": float(threshold),
            "diverged_count": int(np.count_nonzero(has_div)),
            "particle_count": int(len(rows)),
            "relation_counts": {str(k): int(v) for k, v in counts.items()},
            "diverged_solver_state_counts": {str(k): int(v) for k, v in state_counts.items()},
            "diverged_source_part_counts": {str(k): int(v) for k, v in source_counts.items()},
            "first_divergence_time_s": _finite_summary(div_time),
            "first_wall_time_s_for_diverged": _finite_summary(wall_time[has_div]),
        }
    detail = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame()
    detail.to_csv(out_dir / "divergence_alignment.csv", index=False)
    payload = {
        "available": True,
        "thresholds_m": [float(v) for v in thresholds_m],
        "coincidence_window_s": float(coincidence_window_s),
        "particle_count": int(len(particle_ids)),
        "particles_with_wall_event_count": int(np.count_nonzero(np.isfinite(particle_base["first_wall_time_s"].to_numpy(dtype=np.float64))))
        if "first_wall_time_s" in particle_base.columns
        else 0,
        "by_threshold": summary_by_threshold,
        "outputs": {"divergence_alignment_csv": str(out_dir / "divergence_alignment.csv")},
    }
    _write_json(out_dir / "divergence_alignment.json", payload)
    return payload


def _summary_max(summary: Mapping[str, Any], key: str) -> float | None:
    item = summary.get(key, {}) if isinstance(summary, Mapping) else {}
    if not isinstance(item, Mapping):
        return None
    try:
        value = float(item.get("max"))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _comparison_readiness(
    *,
    force_alignment: Mapping[str, Any],
    release_alignment: Mapping[str, Any],
    field_alignment: Mapping[str, Any],
    boundary_role_alignment: Mapping[str, Any],
    finite_boundary_count: int,
) -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []

    for gap in force_alignment.get("force_physics_gaps", []) if isinstance(force_alignment, Mapping) else []:
        if isinstance(gap, Mapping):
            payload = dict(gap)
            if "category" in payload:
                payload["gap_category"] = payload.pop("category")
            blockers.append({"category": "force_physics", "severity": "blocker", **payload})

    matched_release = (
        release_alignment.get("matched_release_errors", {}) if isinstance(release_alignment, Mapping) else {}
    )
    if not bool(matched_release.get("available", False)):
        blockers.append(
            {
                "category": "release_alignment",
                "severity": "blocker",
                "message": "release alignment is missing or did not match particle_id rows",
            }
        )
    else:
        release_pos_max = _summary_max(matched_release, "release_position_error_m")
        if release_pos_max is not None and release_pos_max > 0.0:
            blockers.append(
                {
                    "category": "diagnostic_release",
                    "severity": "blocker",
                    "message": "solver particles are not the exact COMSOL release; this is a diagnostic run",
                    "release_position_error_max_m": release_pos_max,
                }
            )
        if matched_release.get("source_entity_match_ratio") is None:
            blockers.append(
                {
                    "category": "release_source",
                    "severity": "blocker",
                    "message": "row-level COMSOL release source is unavailable, so source parity is not directly checked",
                }
            )

    if not bool(field_alignment.get("available", False)):
        blockers.append({"category": "field_replay", "severity": "blocker", "message": "field replay was not executed"})
    else:
        try:
            support = float(field_alignment.get("support_fraction"))
        except (TypeError, ValueError):
            support = float("nan")
        if not np.isfinite(support) or support < 1.0:
            blockers.append(
                {
                    "category": "field_support",
                    "severity": "blocker",
                    "message": "field replay does not cover all COMSOL trajectory samples",
                    "support_fraction": field_alignment.get("support_fraction"),
                }
            )

    mismatch_count = boundary_role_alignment.get("mismatch_count") if isinstance(boundary_role_alignment, Mapping) else None
    if mismatch_count not in (0, None):
        blockers.append(
            {
                "category": "boundary_roles",
                "severity": "blocker",
                "message": "solver wall laws do not match COMSOL boundary role inventory",
                "mismatch_count": mismatch_count,
            }
        )
    if int(finite_boundary_count) == 0:
        blockers.append(
            {
                "category": "boundary_events",
                "severity": "blocker",
                "message": "COMSOL first-hit entity/normal truth is unavailable; direct boundary parity is blocked",
            }
        )
    return {
        "ready_for_exact_solver_comparison": not blockers,
        "blocker_count": int(len(blockers)),
        "blockers": blockers,
    }


def compare_particle_results(
    *,
    solver_output_dir: str | Path,
    comsol_particle_csv: str | Path,
    out_dir: str | Path,
    boundary_map_csv: str | Path | None = None,
    state_map_json: str | Path | None = None,
    raw_export_dir: str | Path | None = None,
    solver_particles_csv: str | Path | None = None,
    comsol_release_csv: str | Path | None = None,
    comsol_wall_events_csv: str | Path | None = None,
    comsol_particle_status_csv: str | Path | None = None,
    comsol_trajectory_csv: str | Path | None = None,
    field_npz: str | Path | None = None,
    part_walls_csv: str | Path | None = None,
) -> dict[str, Any]:
    solver_dir = Path(solver_output_dir)
    comsol_csv = Path(comsol_particle_csv)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(raw_export_dir) if raw_export_dir else None
    solver_particles_path = Path(solver_particles_csv) if solver_particles_csv else None
    comsol_release_path = Path(comsol_release_csv) if comsol_release_csv else None
    comsol_wall_events_path = Path(comsol_wall_events_csv) if comsol_wall_events_csv else None
    comsol_particle_status_path = Path(comsol_particle_status_csv) if comsol_particle_status_csv else None
    comsol_trajectory_path = Path(comsol_trajectory_csv) if comsol_trajectory_csv else None
    field_npz_path = Path(field_npz) if field_npz else _default_case_file(solver_particles_path, "generated/comsol_field_2d.npz")
    part_walls_path = Path(part_walls_csv) if part_walls_csv else _default_case_file(solver_particles_path, "part_walls.csv")
    boundary_map_path = Path(boundary_map_csv) if boundary_map_csv else _default_case_file(
        solver_particles_path,
        "generated/comsol_boundary_entity_mapping.csv",
    )

    boundary_map = _load_boundary_map(boundary_map_path)
    state_map = _read_json(Path(state_map_json) if state_map_json else None)

    solver = _solver_final_frame(solver_dir).merge(_solver_first_hit_frame(solver_dir), on="particle_id", how="left")
    comsol = _comsol_frame(comsol_csv, boundary_map=boundary_map, state_map=state_map)
    if comsol_particle_status_path is not None and comsol_particle_status_path.exists():
        comsol_status = _comsol_particle_status_frame(comsol_particle_status_path)
        comsol = _merge_comsol_particle_status(comsol, comsol_status, state_map)
    if comsol_wall_events_path is not None and comsol_wall_events_path.exists():
        comsol_events = _comsol_wall_event_frame(comsol_wall_events_path, boundary_map=boundary_map)
        comsol = _merge_comsol_wall_events(comsol, comsol_events)
    matched = solver.merge(comsol, on="particle_id", how="inner")
    if matched.empty:
        raise ValueError("no matching particle_id values between solver and COMSOL results")

    solver_pos = matched[[c for c in matched.columns if c.startswith("solver_final_")]].rename(columns=lambda c: c.replace("solver_final_", ""))
    comsol_pos = matched[[c for c in matched.columns if c.startswith("comsol_final_")]].rename(columns=lambda c: c.replace("comsol_final_", ""))
    solver_hit_pos = matched[[c for c in matched.columns if c.startswith("solver_hit_") and c[-1:] in {"x", "y", "z"}]].rename(columns=lambda c: c.replace("solver_hit_", ""))
    comsol_hit_pos = matched[[c for c in matched.columns if c.startswith("comsol_hit_") and c[-1:] in {"x", "y", "z"}]].rename(columns=lambda c: c.replace("comsol_hit_", ""))
    solver_vel = matched[[c for c in matched.columns if c.startswith("solver_v_")]].rename(columns=lambda c: c.replace("solver_", ""))
    comsol_vel = matched[[c for c in matched.columns if c.startswith("comsol_v_")]].rename(columns=lambda c: c.replace("comsol_", ""))

    matched["state_match"] = matched["solver_state"].astype(str) == matched["comsol_state"].astype(str)
    matched["boundary_match"] = matched["solver_hit_part_id"].to_numpy(dtype=np.float64) == matched["comsol_hit_part_id"].to_numpy(dtype=np.float64)
    matched["hit_time_error_s"] = np.abs(matched["solver_hit_time_s"].to_numpy(dtype=np.float64) - matched["comsol_hit_time_s"].to_numpy(dtype=np.float64))
    if "comsol_status_stop_time_s" not in matched.columns:
        matched["comsol_status_stop_time_s"] = np.nan
    if "comsol_final_status" not in matched.columns:
        matched["comsol_final_status"] = ""
    if "comsol_final_status_code" not in matched.columns:
        matched["comsol_final_status_code"] = np.nan
    matched["solver_first_wall_vs_comsol_stop_time_error_s"] = np.abs(
        matched["solver_hit_time_s"].to_numpy(dtype=np.float64)
        - matched["comsol_status_stop_time_s"].to_numpy(dtype=np.float64)
    )
    matched["final_position_error_m"] = _norm_error(solver_pos, comsol_pos)
    matched["hit_position_error_m"] = _norm_error(solver_hit_pos, comsol_hit_pos)
    matched["final_velocity_error_mps"] = _norm_error(solver_vel, comsol_vel)
    matched["charge_error_C"] = np.abs(matched["solver_charge_C"].to_numpy(dtype=np.float64) - matched["comsol_charge_C"].to_numpy(dtype=np.float64))

    state_counts = _counts_frame(matched["solver_state"], matched["comsol_state"], left_name="solver_count", right_name="comsol_count", key_name="state")
    boundary_counts = _counts_frame(
        matched["solver_hit_part_id"].fillna(-1).astype(int),
        matched["comsol_hit_part_id"].fillna(-1).astype(int),
        left_name="solver_first_hit_count",
        right_name="comsol_first_hit_count",
        key_name="part_id",
    )

    matched_errors = matched[
        [
            "particle_id",
            "solver_state",
            "comsol_state",
            "state_match",
            "solver_hit_part_id",
            "comsol_hit_part_id",
            "boundary_match",
            "solver_hit_outcome",
            "comsol_hit_outcome",
            "hit_time_error_s",
            "comsol_status_stop_time_s",
            "comsol_final_status",
            "comsol_final_status_code",
            "solver_first_wall_vs_comsol_stop_time_error_s",
            "final_position_error_m",
            "hit_position_error_m",
            "final_velocity_error_mps",
            "charge_error_C",
        ]
    ].copy()
    matched_errors.to_csv(out / "matched_particle_errors.csv", index=False)
    state_counts.to_csv(out / "comparison_by_state.csv", index=False)
    boundary_counts.to_csv(out / "comparison_by_boundary.csv", index=False)
    force_alignment = _write_force_alignment(solver_output_dir=solver_dir, raw_export_dir=raw_dir, out_dir=out)
    release_alignment = _write_release_alignment(
        solver_output_dir=solver_dir,
        raw_export_dir=raw_dir,
        out_dir=out,
        solver_particles_csv=solver_particles_path,
        comsol_release_csv=comsol_release_path,
    )
    trajectory_alignment = _write_trajectory_alignment(
        solver_output_dir=solver_dir,
        comsol_trajectory_csv=comsol_trajectory_path,
        out_dir=out,
    )
    field_alignment = _write_field_alignment(
        field_npz=field_npz_path,
        comsol_trajectory_csv=comsol_trajectory_path,
        solver_particles_csv=solver_particles_path,
        out_dir=out,
    )
    trend_alignment = _write_trend_alignment(
        solver_output_dir=solver_dir,
        comsol_trajectory_csv=comsol_trajectory_path,
        solver_particles_csv=solver_particles_path,
        out_dir=out,
    )
    divergence_alignment = _write_divergence_alignment(
        solver_output_dir=solver_dir,
        solver_particles_csv=solver_particles_path,
        out_dir=out,
    )
    if raw_dir is not None and (raw_dir / "physics_feature_inventory.json").exists():
        boundary_role_alignment = derive_boundary_roles(
            raw_export_dir=raw_dir,
            boundary_map_csv=boundary_map_path,
            part_walls_csv=part_walls_path,
            out_dir=out,
        )
    else:
        boundary_role_alignment = {"available": False, "reason": "raw COMSOL physics inventory not provided"}
        _write_json(out / "boundary_role_alignment.json", boundary_role_alignment)

    finite_boundary = np.isfinite(matched["solver_hit_part_id"].to_numpy(dtype=np.float64)) & np.isfinite(matched["comsol_hit_part_id"].to_numpy(dtype=np.float64))
    finite_hit_time = np.isfinite(matched["hit_time_error_s"].to_numpy(dtype=np.float64))
    finite_status_stop = np.isfinite(matched["solver_first_wall_vs_comsol_stop_time_error_s"].to_numpy(dtype=np.float64))
    comparison_readiness = _comparison_readiness(
        force_alignment=force_alignment,
        release_alignment=release_alignment,
        field_alignment=field_alignment,
        boundary_role_alignment=boundary_role_alignment,
        finite_boundary_count=int(np.count_nonzero(finite_boundary)),
    )
    _write_json(out / "comparison_readiness.json", comparison_readiness)

    summary = {
        "solver_output_dir": str(solver_dir),
        "comsol_particle_csv": str(comsol_csv),
        "solver_particle_count": int(len(solver)),
        "comsol_particle_count": int(len(comsol)),
        "matched_particle_count": int(len(matched)),
        "state_match_ratio": float(matched["state_match"].mean()),
        "first_hit_boundary_match_ratio": float(matched.loc[finite_boundary, "boundary_match"].mean()) if np.any(finite_boundary) else None,
        "first_hit_time_comparison_count": int(np.count_nonzero(finite_hit_time)),
        "hit_time_error_s": _finite_summary(matched["hit_time_error_s"].to_numpy(dtype=np.float64)),
        "particle_status_stop_time_comparison_count": int(np.count_nonzero(finite_status_stop)),
        "solver_first_wall_vs_comsol_stop_time_error_s": _finite_summary(
            matched["solver_first_wall_vs_comsol_stop_time_error_s"].to_numpy(dtype=np.float64)
        ),
        "final_position_error_m": _finite_summary(matched["final_position_error_m"].to_numpy(dtype=np.float64)),
        "hit_position_error_m": _finite_summary(matched["hit_position_error_m"].to_numpy(dtype=np.float64)),
        "final_velocity_error_mps": _finite_summary(matched["final_velocity_error_mps"].to_numpy(dtype=np.float64)),
        "charge_error_C": _finite_summary(matched["charge_error_C"].to_numpy(dtype=np.float64)),
        "comparison_readiness": comparison_readiness,
        "force_model_alignment": {
            "solver_enabled_forces": force_alignment.get("solver_enabled_forces", []),
            "comsol_force_kinds": force_alignment.get("comsol_force_kinds", []),
            "comsol_non_particle_force_kinds": force_alignment.get("comsol_non_particle_force_kinds", []),
            "comsol_force_without_enabled_solver_counterpart": force_alignment.get(
                "comsol_force_without_enabled_solver_counterpart",
                [],
            ),
            "force_physics_gaps": force_alignment.get("force_physics_gaps", []),
            "force_contribution_alignment": force_alignment.get("force_contribution_alignment", []),
            "force_contribution_alignment_csv": force_alignment.get("force_contribution_alignment_csv", ""),
            "enabled_solver_force_missing_export_field": force_alignment.get(
                "enabled_solver_force_missing_export_field",
                [],
            ),
        },
        "release_alignment": {
            "comsol_release_feature_count": release_alignment.get("comsol_release_feature_count", 0),
            "comsol_release_kinds": release_alignment.get("comsol_release_kinds", []),
            "solver_particles_available": bool(release_alignment.get("solver_particles", {}).get("available", False)),
            "comsol_release_particles_available": bool(
                release_alignment.get("comsol_release_particles", {}).get("available", False)
            ),
            "matched_release_errors": release_alignment.get("matched_release_errors", {}),
        },
        "outputs": {
            "summary_json": str(out / "comparison_summary.json"),
            "by_state_csv": str(out / "comparison_by_state.csv"),
            "by_boundary_csv": str(out / "comparison_by_boundary.csv"),
            "matched_particle_errors_csv": str(out / "matched_particle_errors.csv"),
            "force_model_alignment_json": str(out / "force_model_alignment.json"),
            "release_alignment_json": str(out / "release_alignment.json"),
            "trajectory_alignment_json": str(out / "trajectory_alignment.json"),
            "matched_trajectory_errors_csv": str(out / "matched_trajectory_errors.csv"),
            "distribution_alignment_csv": str(out / "distribution_alignment.csv"),
            "field_alignment_json": str(out / "field_alignment.json"),
            "boundary_role_alignment_json": str(out / "boundary_role_alignment.json"),
            "trend_alignment_json": str(out / "trend_alignment.json"),
            "divergence_alignment_json": str(out / "divergence_alignment.json"),
            "comparison_readiness_json": str(out / "comparison_readiness.json"),
            "comsol_wall_events_csv": str(comsol_wall_events_path) if comsol_wall_events_path is not None else "",
            "comsol_particle_status_csv": str(comsol_particle_status_path)
            if comsol_particle_status_path is not None
            else "",
        },
        "trajectory_alignment": {
            "available": bool(trajectory_alignment.get("available", False)),
            "matched_sample_count": trajectory_alignment.get("matched_sample_count", 0),
            "matched_particle_count": trajectory_alignment.get("matched_particle_count", 0),
            "distribution_time_count": trajectory_alignment.get("distribution_time_count", 0),
        },
        "field_alignment": {
            "available": bool(field_alignment.get("available", False)),
            "field_backend_kind": field_alignment.get("field_backend_kind"),
            "sample_count": field_alignment.get("sample_count", 0),
            "clean_stencil_fraction": field_alignment.get("clean_stencil_fraction"),
            "support_fraction": field_alignment.get("support_fraction"),
            "velocity_residual_mps": field_alignment.get("velocity_residual_mps", {}),
        },
        "boundary_role_alignment": {
            "available": bool(boundary_role_alignment.get("available", False)),
            "mismatch_count": boundary_role_alignment.get("mismatch_count", 0),
            "expected_role_counts": boundary_role_alignment.get("expected_role_counts", {}),
        },
        "trend_alignment": {
            "available": bool(trend_alignment.get("available", False)),
            "comsol_finite_at_final_count": trend_alignment.get("comsol_finite_at_final_count", 0),
            "solver_state_counts": trend_alignment.get("solver_state_counts", {}),
        },
        "divergence_alignment": {
            "available": bool(divergence_alignment.get("available", False)),
            "particles_with_wall_event_count": divergence_alignment.get("particles_with_wall_event_count", 0),
            "by_threshold": divergence_alignment.get("by_threshold", {}),
        },
    }
    _write_json(out / "comparison_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare external COMSOL particle tracing results with a solver output directory.")
    parser.add_argument("--solver-output-dir", type=Path, required=True)
    parser.add_argument("--comsol-particle-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--boundary-map-csv", type=Path, default=None)
    parser.add_argument("--state-map-json", type=Path, default=None)
    parser.add_argument("--raw-export-dir", type=Path, default=None)
    parser.add_argument("--solver-particles-csv", type=Path, default=None)
    parser.add_argument("--comsol-release-csv", type=Path, default=None)
    parser.add_argument("--comsol-wall-events-csv", type=Path, default=None)
    parser.add_argument("--comsol-particle-status-csv", type=Path, default=None)
    parser.add_argument("--comsol-trajectory-csv", type=Path, default=None)
    parser.add_argument("--field-npz", type=Path, default=None)
    parser.add_argument("--part-walls-csv", type=Path, default=None)
    args = parser.parse_args(argv)

    summary = compare_particle_results(
        solver_output_dir=args.solver_output_dir,
        comsol_particle_csv=args.comsol_particle_csv,
        out_dir=args.out_dir,
        boundary_map_csv=args.boundary_map_csv,
        state_map_json=args.state_map_json,
        raw_export_dir=args.raw_export_dir,
        solver_particles_csv=args.solver_particles_csv,
        comsol_release_csv=args.comsol_release_csv,
        comsol_wall_events_csv=args.comsol_wall_events_csv,
        comsol_particle_status_csv=args.comsol_particle_status_csv,
        comsol_trajectory_csv=args.comsol_trajectory_csv,
        field_npz=args.field_npz,
        part_walls_csv=args.part_walls_csv,
    )
    print(json.dumps(summary, indent=2))
    return 0
