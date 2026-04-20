from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


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
    left = _first_column(frame, ("comsol_boundary_id", "comsol_entity_id", "boundary_id", "hit_boundary_id"))
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
    out["comsol_hit_time_s"] = _numeric_or_nan(raw, ("hit_time_s", "hit_time", "t_hit", "wall_time"))
    raw_boundary = _numeric_or_nan(raw, ("hit_boundary_id", "boundary_id", "hit_boundary", "part_id"))
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
        out.append(
            {
                "component_tag": item.get("component_tag", ""),
                "physics_tag": item.get("physics_tag", ""),
                "feature_tag": item.get("feature_tag", ""),
                "label": item.get("label", ""),
                "type": item.get("type", ""),
                "force_kind": kind,
                "selection_entities": item.get("selection_entities", []),
            }
        )
    return out


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


def _write_force_alignment(
    *,
    solver_output_dir: Path,
    raw_export_dir: Path | None,
    out_dir: Path,
) -> dict[str, Any]:
    solver_runtime, solver_catalog = _solver_force_payloads(solver_output_dir)
    solver_enabled = _enabled_solver_force_names(solver_runtime, solver_catalog)
    comsol_features = _force_features_from_raw_export(raw_export_dir)
    comsol_kinds = sorted({str(item.get("force_kind", "")) for item in comsol_features if item.get("force_kind")})
    fields = _selected_force_fields(raw_export_dir)
    missing_solver = [name for name in comsol_kinds if name not in solver_enabled]
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
        "selected_force_fields": fields,
        "comsol_force_without_enabled_solver_counterpart": missing_solver,
        "enabled_solver_force_missing_export_field": sorted(set(missing_export_fields)),
    }
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
    matched = s.merge(c, on="_particle_id", suffixes=("_solver", "_comsol"))
    if matched.empty:
        return {"available": False, "reason": "no matched particle_id"}
    solver_time = _release_time_values(matched.rename(columns={col: col.replace("_solver", "") for col in matched.columns if col.endswith("_solver")}))
    comsol_time = _release_time_values(matched.rename(columns={col: col.replace("_comsol", "") for col in matched.columns if col.endswith("_comsol")}))
    return {
        "available": True,
        "matched_particle_count": int(len(matched)),
        "release_time_error_s": _finite_summary(np.abs(solver_time - comsol_time)),
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
) -> dict[str, Any]:
    solver_dir = Path(solver_output_dir)
    comsol_csv = Path(comsol_particle_csv)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(raw_export_dir) if raw_export_dir else None
    solver_particles_path = Path(solver_particles_csv) if solver_particles_csv else None
    comsol_release_path = Path(comsol_release_csv) if comsol_release_csv else None

    boundary_map = _load_boundary_map(Path(boundary_map_csv) if boundary_map_csv else None)
    state_map = _read_json(Path(state_map_json) if state_map_json else None)

    solver = _solver_final_frame(solver_dir).merge(_solver_first_hit_frame(solver_dir), on="particle_id", how="left")
    comsol = _comsol_frame(comsol_csv, boundary_map=boundary_map, state_map=state_map)
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
            "hit_time_error_s",
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

    finite_boundary = np.isfinite(matched["solver_hit_part_id"].to_numpy(dtype=np.float64)) & np.isfinite(matched["comsol_hit_part_id"].to_numpy(dtype=np.float64))
    summary = {
        "solver_output_dir": str(solver_dir),
        "comsol_particle_csv": str(comsol_csv),
        "solver_particle_count": int(len(solver)),
        "comsol_particle_count": int(len(comsol)),
        "matched_particle_count": int(len(matched)),
        "state_match_ratio": float(matched["state_match"].mean()),
        "first_hit_boundary_match_ratio": float(matched.loc[finite_boundary, "boundary_match"].mean()) if np.any(finite_boundary) else None,
        "hit_time_error_s": _finite_summary(matched["hit_time_error_s"].to_numpy(dtype=np.float64)),
        "final_position_error_m": _finite_summary(matched["final_position_error_m"].to_numpy(dtype=np.float64)),
        "hit_position_error_m": _finite_summary(matched["hit_position_error_m"].to_numpy(dtype=np.float64)),
        "final_velocity_error_mps": _finite_summary(matched["final_velocity_error_mps"].to_numpy(dtype=np.float64)),
        "charge_error_C": _finite_summary(matched["charge_error_C"].to_numpy(dtype=np.float64)),
        "force_model_alignment": {
            "solver_enabled_forces": force_alignment.get("solver_enabled_forces", []),
            "comsol_force_kinds": force_alignment.get("comsol_force_kinds", []),
            "comsol_force_without_enabled_solver_counterpart": force_alignment.get(
                "comsol_force_without_enabled_solver_counterpart",
                [],
            ),
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
        },
        "outputs": {
            "summary_json": str(out / "comparison_summary.json"),
            "by_state_csv": str(out / "comparison_by_state.csv"),
            "by_boundary_csv": str(out / "comparison_by_boundary.csv"),
            "matched_particle_errors_csv": str(out / "matched_particle_errors.csv"),
            "force_model_alignment_json": str(out / "force_model_alignment.json"),
            "release_alignment_json": str(out / "release_alignment.json"),
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
    )
    print(json.dumps(summary, indent=2))
    return 0
