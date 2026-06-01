from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


COUNTER_KEYS = (
    "unresolved_crossing_count",
    "max_hits_reached_count",
    "nearest_projection_fallback_count",
    "bisection_fallback_count",
    "numerical_boundary_stopped_count",
    "boundary_event_failure_count",
    "invalid_mask_stopped_count",
    "valid_mask_mixed_stencil_count",
    "valid_mask_hard_invalid_count",
    "source_surface_release_skip_count",
    "source_surface_release_skip_blocked_count",
)

PARTICLE_ID_ALIASES = ("particle_id", "ParticleID", "particle", "pid", "id")
TIME_ALIASES = ("time_s", "time", "t", "Time")
SOURCE_PART_ALIASES = (
    "source_part_id",
    "source_entity_id",
    "release_part_id",
    "release_entity_id",
    "part_id",
    "entity_id",
)
BOUNDARY_PART_ALIASES = (
    "part_id",
    "boundary_part_id",
    "wall_part_id",
    "comsol_entity_id",
    "entity_id",
    "boundary_id",
)
STATE_ALIASES = ("particle_class", "final_state", "state", "status", "outcome")
POSITION_ALIASES = (
    ("x", "r", "x_m", "r_m", "position_x"),
    ("y", "z", "y_m", "z_m", "position_y"),
    ("z", "z_m", "position_z"),
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _finite_summary(values: Sequence[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50.0)),
        "p90": float(np.percentile(finite, 90.0)),
        "max": float(np.max(finite)),
    }


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {"value": payload}


def _read_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _artifact(name: str, path: Path | None, *, required: bool = False) -> dict[str, Any]:
    exists = bool(path is not None and path.exists())
    return {
        "name": name,
        "path": "" if path is None else str(path),
        "status": "found" if exists else ("missing_required" if required else "missing_optional"),
        "required": int(bool(required)),
    }


def _discover(root: Path | None, filename: str) -> Path | None:
    if root is None:
        return None
    direct = root / filename
    return direct if direct.exists() else None


def _find_column(frame: pd.DataFrame | None, aliases: Sequence[str]) -> str | None:
    if frame is None:
        return None
    lower = {str(col).strip().lower(): str(col) for col in frame.columns}
    for alias in aliases:
        found = lower.get(str(alias).strip().lower())
        if found is not None:
            return found
    return None


def _numeric_column(frame: pd.DataFrame, aliases: Sequence[str], *, default: float = np.nan) -> pd.Series:
    col = _find_column(frame, aliases)
    if col is None:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce")


def _text_column(frame: pd.DataFrame, aliases: Sequence[str]) -> pd.Series:
    col = _find_column(frame, aliases)
    if col is None:
        return pd.Series("", index=frame.index, dtype=object)
    return frame[col].fillna("").astype(str)


def _particle_ids(frame: pd.DataFrame, *, label: str) -> pd.Series:
    col = _find_column(frame, PARTICLE_ID_ALIASES)
    if col is None:
        raise ValueError(f"{label} CSV must contain a particle_id column")
    values = pd.to_numeric(frame[col], errors="coerce")
    return values.astype("Int64")


def _final_rows(long_frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    ids = _particle_ids(long_frame, label=label)
    work = long_frame.copy()
    work["_particle_id"] = ids
    work = work[work["_particle_id"].notna()].copy()
    if work.empty:
        return work
    time_col = _find_column(work, TIME_ALIASES)
    if time_col is not None:
        work["_time_sort"] = pd.to_numeric(work[time_col], errors="coerce").fillna(-np.inf)
        work = work.sort_values(["_particle_id", "_time_sort"], kind="mergesort")
    return work.groupby("_particle_id", as_index=False, sort=False).tail(1).copy()


def _canonical_state_frame(frame: pd.DataFrame, *, label: str, is_solver: bool) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=["particle_id", "state_class"])
    ids = _particle_ids(frame, label=label)
    states = pd.Series("unknown", index=frame.index, dtype=object)
    if is_solver:
        for column, state in (("stuck", "stuck"), ("absorbed", "absorbed"), ("escaped", "vacuum"), ("vacuum", "vacuum")):
            if column in frame.columns:
                mask = pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy(dtype=float) != 0.0
                states.loc[mask] = state
        if "active" in frame.columns:
            active = pd.to_numeric(frame["active"], errors="coerce").fillna(0).to_numpy(dtype=float) != 0.0
            states.loc[active & (states == "unknown")] = "active"
    else:
        text = _text_column(frame, STATE_ALIASES).str.lower()
        states.loc[text.str.contains("stuck|deposit|wall", regex=True, na=False)] = "stuck"
        states.loc[text.str.contains("absor", regex=True, na=False)] = "absorbed"
        states.loc[text.str.contains("escape|vacuum|outlet|lost", regex=True, na=False)] = "vacuum"
        states.loc[text.str.contains("active|free|running", regex=True, na=False)] = "active"
        for column, state in (("stuck", "stuck"), ("absorbed", "absorbed"), ("escaped", "vacuum"), ("vacuum", "vacuum"), ("active", "active")):
            if column in frame.columns:
                mask = pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy(dtype=float) != 0.0
                states.loc[mask] = state
    out = pd.DataFrame({"particle_id": ids, "state_class": states})
    return out[out["particle_id"].notna()].copy()


def _state_summary(frame: pd.DataFrame | None, *, label: str, is_solver: bool) -> dict[str, Any]:
    if frame is None:
        return {"available": 0, "particle_count": 0, "counts": {}, "fractions": {}, "escaped_or_vacuum_fraction": None}
    states = _canonical_state_frame(frame, label=label, is_solver=is_solver)
    count = int(len(states))
    counts = {str(k): int(v) for k, v in states["state_class"].astype(str).value_counts().items()}
    fractions = {key: (float(value) / float(count) if count > 0 else None) for key, value in counts.items()}
    return {
        "available": 1,
        "metric_scope": "final_snapshot",
        "particle_count": count,
        "counts": counts,
        "fractions": fractions,
        "escaped_or_vacuum_fraction": fractions.get("vacuum"),
    }


def _position_frame(frame: pd.DataFrame | None, *, label: str) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=["particle_id"])
    ids = _particle_ids(frame, label=label)
    out = pd.DataFrame({"particle_id": ids})
    for axis_index, aliases in enumerate(POSITION_ALIASES):
        col = _find_column(frame, aliases)
        if col is not None:
            out[f"pos_{axis_index}"] = pd.to_numeric(frame[col], errors="coerce")
    return out[out["particle_id"].notna()].copy()


def _source_part_frame(frame: pd.DataFrame | None, *, label: str) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(columns=["particle_id", "source_part_id"])
    ids = _particle_ids(frame, label=label)
    parts = _numeric_column(frame, SOURCE_PART_ALIASES, default=0.0).fillna(0).astype(int)
    out = pd.DataFrame({"particle_id": ids, "source_part_id": parts})
    out = out[out["particle_id"].notna()].copy()
    return out.drop_duplicates("particle_id", keep="last")


def _event_particle_ids(frame: pd.DataFrame | None, *, label: str, is_solver: bool) -> set[int]:
    if frame is None or frame.empty:
        return set()
    ids = _particle_ids(frame, label=label)
    if is_solver:
        return {int(value) for value in ids.dropna().to_numpy(dtype=np.int64)}
    event_mask = pd.Series(False, index=frame.index)
    for column in ("wall_hit", "wallhit", "hit", "event", "has_wall_hit"):
        if column in frame.columns:
            event_mask = event_mask | (pd.to_numeric(frame[column], errors="coerce").fillna(0) != 0)
    part_col = _find_column(frame, BOUNDARY_PART_ALIASES)
    if part_col is not None:
        part_values = pd.to_numeric(frame[part_col], errors="coerce")
        event_mask = event_mask | (np.isfinite(part_values.to_numpy(dtype=float)) & (part_values.to_numpy(dtype=float) > 0.0))
    text = _text_column(frame, ("event_type", "event", "outcome", "status", "state")).str.lower()
    event_mask = event_mask | text.str.contains("wall|hit|stuck|deposit|absor|bounce|reflect", regex=True, na=False)
    return {int(value) for value in ids[event_mask & ids.notna()].to_numpy(dtype=np.int64)}


def _event_summary(
    *,
    particle_count: int,
    solver_events: set[int],
    comsol_events: set[int],
    solver_events_available: bool,
    comsol_events_available: bool,
) -> dict[str, Any]:
    solver_count = int(len(solver_events))
    comsol_count = int(len(comsol_events))
    denom = float(particle_count) if particle_count > 0 else 0.0
    return {
        "metric_scope": "ever_reached_wall_event_proxy_when_artifacts_available",
        "solver_events_available": int(bool(solver_events_available)),
        "comsol_events_available": int(bool(comsol_events_available)),
        "solver_wallhit_particle_count": solver_count,
        "comsol_wallhit_particle_count": comsol_count,
        "zero_wallhit_fraction_solver": None if denom == 0.0 or not solver_events_available else float((particle_count - solver_count) / denom),
        "zero_wallhit_fraction_comsol": None if denom == 0.0 or not comsol_events_available else float((particle_count - comsol_count) / denom),
        "solver_only_event_count": int(len(solver_events.difference(comsol_events))),
        "comsol_only_event_count": int(len(comsol_events.difference(solver_events))),
    }


def _summarize_preprocess(source_diag: pd.DataFrame | None, solver_report: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": int(source_diag is not None),
        "metric_scope": "post_preprocess_source_diagnostics_when_available",
    }
    if source_diag is None:
        out["particle_count"] = int(solver_report.get("particle_count", 0) or 0) if isinstance(solver_report, Mapping) else 0
        return out
    count = int(len(source_diag))
    applied = int(pd.to_numeric(source_diag.get("boundary_release_applied", pd.Series(0, index=source_diag.index)), errors="coerce").fillna(0).sum())
    failed = int(pd.to_numeric(source_diag.get("boundary_release_failed_offset", pd.Series(0, index=source_diag.index)), errors="coerce").fillna(0).sum())
    out.update(
        {
            "particle_count": count,
            "boundary_release_applied_count": applied,
            "boundary_release_applied_ratio": float(applied / count) if count > 0 else None,
            "boundary_release_failed_offset_count": failed,
            "boundary_release_failed_offset_ratio": float(failed / count) if count > 0 else None,
        }
    )
    for column in ("projection_distance_m", "boundary_release_projection_distance_m"):
        if column in source_diag.columns:
            out["projection_distance_m"] = _finite_summary(pd.to_numeric(source_diag[column], errors="coerce").to_numpy())
            break
    if "source_provenance_group" in source_diag.columns:
        out["source_provenance_counts"] = {
            str(k): int(v)
            for k, v in source_diag["source_provenance_group"].fillna("unknown_source").astype(str).value_counts().items()
        }
    return out


def _summarize_first_step(first_step_dir: Path | None) -> tuple[dict[str, Any], Path | None]:
    if first_step_dir is None:
        return {"available": 0}, None
    summary_path = _discover(first_step_dir, "first_step_compare_summary.json") or _discover(first_step_dir, "first_step_summary.json")
    error_path = _discover(first_step_dir, "first_step_error.csv")
    summary = _read_json(summary_path)
    error = _read_csv(error_path)
    out: dict[str, Any] = {
        "available": int(summary is not None or error is not None),
        "metric_scope": "post_preprocess_to_post_first_step",
    }
    if isinstance(summary, Mapping):
        out.update(
            {
                "particle_count": summary.get("particle_count"),
                "compared_particle_count": summary.get("compared_particle_count"),
                "stochastic_policy": summary.get("stochastic_policy"),
                "position_error_m": summary.get("position_error_m", {}),
                "velocity_error_mps": summary.get("velocity_error_mps", {}),
            }
        )
    if error is not None and not error.empty:
        for column in ("position_error_m", "velocity_error_mps", "speed_ratio"):
            if column in error.columns:
                out[column] = _finite_summary(pd.to_numeric(error[column], errors="coerce").to_numpy())
    return out, error_path


def _near_wall_active_summary(solver_report: Mapping[str, Any] | None, collision_diag: Mapping[str, Any] | None) -> dict[str, Any]:
    source: dict[str, Any] = {}
    if isinstance(solver_report, Mapping):
        source.update(dict(solver_report))
    if isinstance(collision_diag, Mapping):
        source.update(dict(collision_diag))
    state_summary = source.get("state_geometry_summary", {}) if isinstance(source, Mapping) else {}
    by_state = state_summary.get("by_state", {}) if isinstance(state_summary, Mapping) else {}
    active: Mapping[str, Any] | None = None
    for key in ("active_free_flight", "active"):
        value = by_state.get(key) if isinstance(by_state, Mapping) else None
        if isinstance(value, Mapping):
            active = value
            break
    return {
        "available": int(active is not None),
        "metric_scope": "final_snapshot_active_near_boundary_proxy_not_ever_reached",
        "near_boundary_threshold_m": state_summary.get("near_boundary_threshold_m") if isinstance(state_summary, Mapping) else None,
        "active_count": None if active is None else int(active.get("count", 0) or 0),
        "active_near_boundary_count": None if active is None else int(active.get("near_boundary_count", 0) or 0),
        "active_nearest_part_counts": [] if active is None else active.get("nearest_part_counts", []),
    }


def _runtime_counters(solver_report: Mapping[str, Any] | None, collision_diag: Mapping[str, Any] | None) -> dict[str, Any]:
    source: dict[str, Any] = {}
    if isinstance(solver_report, Mapping):
        source.update(dict(solver_report))
    if isinstance(collision_diag, Mapping):
        source.update(dict(collision_diag))
    timing = solver_report.get("timing_s", {}) if isinstance(solver_report, Mapping) else {}
    return {
        "timing_s": dict(timing) if isinstance(timing, Mapping) else {},
        "counters": {key: int(source.get(key, 0) or 0) for key in COUNTER_KEYS},
    }


def _final_residual_frame(
    *,
    solver_final: pd.DataFrame | None,
    comsol_final: pd.DataFrame,
    release_reference: pd.DataFrame,
    solver_events: set[int],
    comsol_events: set[int],
) -> pd.DataFrame:
    solver_states = _canonical_state_frame(solver_final, label="solver final_particles", is_solver=True).rename(
        columns={"state_class": "solver_state"}
    )
    comsol_states = _canonical_state_frame(comsol_final, label="COMSOL trajectory", is_solver=False).rename(
        columns={"state_class": "comsol_state"}
    )
    solver_pos = _position_frame(solver_final, label="solver final_particles").add_suffix("_solver")
    comsol_pos = _position_frame(comsol_final, label="COMSOL trajectory").add_suffix("_comsol")
    solver_pos = solver_pos.rename(columns={"particle_id_solver": "particle_id"})
    comsol_pos = comsol_pos.rename(columns={"particle_id_comsol": "particle_id"})
    source_parts = _source_part_frame(release_reference, label="COMSOL release/reference")
    if solver_final is not None and "source_part_id" in solver_final.columns:
        solver_source = _source_part_frame(solver_final, label="solver final_particles").rename(columns={"source_part_id": "solver_source_part_id"})
        source_parts = source_parts.merge(solver_source, on="particle_id", how="outer")
        source_parts["source_part_id"] = source_parts["source_part_id"].fillna(source_parts["solver_source_part_id"]).fillna(0).astype(int)
        source_parts = source_parts[["particle_id", "source_part_id"]]
    merged = comsol_states.merge(solver_states, on="particle_id", how="outer")
    merged = merged.merge(source_parts, on="particle_id", how="left")
    merged = merged.merge(solver_pos, on="particle_id", how="left")
    merged = merged.merge(comsol_pos, on="particle_id", how="left")
    axes = [
        index
        for index in range(3)
        if f"pos_{index}_solver" in merged.columns and f"pos_{index}_comsol" in merged.columns
    ]
    if axes:
        solver_values = merged[[f"pos_{idx}_solver" for idx in axes]].to_numpy(dtype=np.float64)
        comsol_values = merged[[f"pos_{idx}_comsol" for idx in axes]].to_numpy(dtype=np.float64)
        valid = np.all(np.isfinite(solver_values), axis=1) & np.all(np.isfinite(comsol_values), axis=1)
        error = np.full(len(merged), np.nan, dtype=np.float64)
        error[valid] = np.linalg.norm(solver_values[valid] - comsol_values[valid], axis=1)
        merged["final_position_error_m"] = error
    else:
        merged["final_position_error_m"] = np.nan
    known_comsol = merged["comsol_state"].fillna("unknown").astype(str) != "unknown"
    known_solver = merged["solver_state"].fillna("unknown").astype(str) != "unknown"
    merged["final_state_mismatch"] = (
        known_comsol
        & known_solver
        & (merged["comsol_state"].fillna("unknown").astype(str) != merged["solver_state"].fillna("unknown").astype(str))
    )
    merged["solver_only_event"] = merged["particle_id"].astype("Int64").isin(solver_events.difference(comsol_events))
    merged["comsol_only_event"] = merged["particle_id"].astype("Int64").isin(comsol_events.difference(solver_events))
    merged["source_part_id"] = merged["source_part_id"].fillna(0).astype(int)
    return merged


def _top_source_parts(residuals: pd.DataFrame, *, top_n: int = 10) -> list[dict[str, Any]]:
    if residuals.empty:
        return []
    suspicious = residuals[
        residuals["final_state_mismatch"].fillna(False)
        | residuals["solver_only_event"].fillna(False)
        | residuals["comsol_only_event"].fillna(False)
        | (pd.to_numeric(residuals["final_position_error_m"], errors="coerce").fillna(0.0) > 0.0)
    ].copy()
    if suspicious.empty:
        return []
    rows: list[dict[str, Any]] = []
    for part_id, group in suspicious.groupby("source_part_id", dropna=False):
        values = pd.to_numeric(group["final_position_error_m"], errors="coerce").to_numpy(dtype=np.float64)
        rows.append(
            {
                "source_part_id": int(part_id),
                "residual_particle_count": int(len(group)),
                "final_state_mismatch_count": int(group["final_state_mismatch"].fillna(False).sum()),
                "solver_only_event_count": int(group["solver_only_event"].fillna(False).sum()),
                "comsol_only_event_count": int(group["comsol_only_event"].fillna(False).sum()),
                "final_position_error_m": _finite_summary(values),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            int(item["residual_particle_count"]),
            int(item["final_state_mismatch_count"]) + int(item["solver_only_event_count"]) + int(item["comsol_only_event_count"]),
        ),
        reverse=True,
    )[: int(top_n)]


def _write_suspicious_particles(residuals: pd.DataFrame, output_dir: Path, *, limit: int) -> Path | None:
    if int(limit) <= 0 or residuals.empty:
        return None
    suspicious = residuals[
        residuals["final_state_mismatch"].fillna(False)
        | residuals["solver_only_event"].fillna(False)
        | residuals["comsol_only_event"].fillna(False)
        | (pd.to_numeric(residuals["final_position_error_m"], errors="coerce").fillna(0.0) > 0.0)
    ].copy()
    if suspicious.empty:
        return None
    suspicious["residual_rank_score"] = (
        suspicious["final_state_mismatch"].fillna(False).astype(int)
        + suspicious["solver_only_event"].fillna(False).astype(int)
        + suspicious["comsol_only_event"].fillna(False).astype(int)
        + pd.to_numeric(suspicious["final_position_error_m"], errors="coerce").fillna(0.0)
    )
    suspicious = suspicious.sort_values("residual_rank_score", ascending=False).head(int(limit))
    output_path = output_dir / "suspicious_particles.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    keep = [
        "particle_id",
        "source_part_id",
        "solver_state",
        "comsol_state",
        "final_state_mismatch",
        "final_position_error_m",
        "solver_only_event",
        "comsol_only_event",
    ]
    suspicious[[col for col in keep if col in suspicious.columns]].to_csv(output_path, index=False)
    return output_path


def build_full_diagnostics_summary(
    *,
    solver_output_dir: Path,
    comsol_trajectory_csv: Path,
    comsol_release_csv: Path,
    reference_scope: str = "unspecified",
    first_step_dir: Path | None = None,
    suspicious_limit: int = 100,
    output_dir: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, Path | None]:
    solver_output_dir = Path(solver_output_dir)
    first_step_dir = first_step_dir or solver_output_dir
    output_dir = Path(output_dir) if output_dir is not None else Path("comsol_full_diagnostics")

    solver_report_path = _discover(solver_output_dir, "solver_report.json")
    collision_diag_path = _discover(solver_output_dir, "collision_diagnostics.json")
    final_particles_path = _discover(solver_output_dir, "final_particles.csv")
    wall_events_path = _discover(solver_output_dir, "wall_events.csv")
    source_diag_path = _discover(solver_output_dir, "source_particle_diagnostics.csv")
    first_step_summary_path = _discover(first_step_dir, "first_step_compare_summary.json") or _discover(first_step_dir, "first_step_summary.json")
    first_step_error_path = _discover(first_step_dir, "first_step_error.csv")

    artifacts = [
        _artifact("solver_report", solver_report_path, required=True),
        _artifact("final_particles", final_particles_path, required=True),
        _artifact("collision_diagnostics", collision_diag_path),
        _artifact("wall_events", wall_events_path),
        _artifact("source_particle_diagnostics", source_diag_path),
        _artifact("first_step_compare_summary", first_step_summary_path),
        _artifact("first_step_error", first_step_error_path),
        _artifact("comsol_long_trajectory", comsol_trajectory_csv, required=True),
        _artifact("comsol_release_reference", comsol_release_csv, required=True),
    ]

    solver_report = _read_json(solver_report_path)
    collision_diag = _read_json(collision_diag_path)
    solver_final = _read_csv(final_particles_path)
    wall_events = _read_csv(wall_events_path)
    source_diag = _read_csv(source_diag_path)
    comsol_trajectory = _read_csv(comsol_trajectory_csv)
    comsol_release = _read_csv(comsol_release_csv)
    if comsol_trajectory is None:
        raise FileNotFoundError(comsol_trajectory_csv)
    if comsol_release is None:
        raise FileNotFoundError(comsol_release_csv)

    comsol_final = _final_rows(comsol_trajectory, label="COMSOL trajectory")
    full_ids = set(_particle_ids(comsol_trajectory, label="COMSOL trajectory").dropna().to_numpy(dtype=np.int64))
    sampled_ids = set(_particle_ids(comsol_release, label="COMSOL release/reference").dropna().to_numpy(dtype=np.int64))
    solver_particle_count = int(solver_report.get("particle_count", 0) or 0) if isinstance(solver_report, Mapping) else 0
    if solver_final is not None and solver_particle_count <= 0:
        solver_particle_count = int(len(solver_final))

    solver_events = _event_particle_ids(wall_events, label="solver wall_events", is_solver=True)
    comsol_events = _event_particle_ids(comsol_trajectory, label="COMSOL trajectory", is_solver=False)
    residuals = _final_residual_frame(
        solver_final=solver_final,
        comsol_final=comsol_final,
        release_reference=comsol_release,
        solver_events=solver_events,
        comsol_events=comsol_events,
    )
    first_step_summary, _ = _summarize_first_step(first_step_dir)

    summary: dict[str, Any] = {
        "summary_schema_version": 1,
        "reference_scope": str(reference_scope),
        "reference_scope_note": (
            "sampled/full scope is operator-declared; particle counts are reported separately and are not inferred as acceptance"
        ),
        "inputs": {
            "solver_output_dir": str(solver_output_dir),
            "comsol_long_trajectory_csv": str(comsol_trajectory_csv),
            "comsol_release_reference_csv": str(comsol_release_csv),
            "first_step_dir": str(first_step_dir),
        },
        "artifact_status": artifacts,
        "missing_optional_artifacts": [item["name"] for item in artifacts if item["status"] == "missing_optional"],
        "missing_required_artifacts": [item["name"] for item in artifacts if item["status"] == "missing_required"],
        "comsol_reference_counts": {
            "full_trajectory_row_count": int(len(comsol_trajectory)),
            "full_particle_count": int(len(full_ids)),
            "sampled_release_row_count": int(len(comsol_release)),
            "sampled_particle_count": int(len(sampled_ids)),
            "sampled_particles_present_in_full_count": int(len(full_ids.intersection(sampled_ids))),
        },
        "final_state": {
            "metric_scope": "final_snapshot",
            "solver": _state_summary(solver_final, label="solver final_particles", is_solver=True),
            "comsol": _state_summary(comsol_final, label="COMSOL trajectory", is_solver=False),
            "matched_particle_count": int(len(set(residuals["particle_id"].dropna().astype(int)).intersection(full_ids))),
        },
        "preprocess": _summarize_preprocess(source_diag, solver_report),
        "first_step": first_step_summary,
        "near_wall_active": _near_wall_active_summary(solver_report, collision_diag),
        "wall_events": _event_summary(
            particle_count=max(solver_particle_count, len(full_ids)),
            solver_events=solver_events,
            comsol_events=comsol_events,
            solver_events_available=wall_events is not None,
            comsol_events_available=comsol_trajectory is not None,
        ),
        "top_source_parts_for_residuals": _top_source_parts(residuals),
        "runtime_collision_counters": _runtime_counters(solver_report, collision_diag),
    }

    suspicious_path = _write_suspicious_particles(residuals, output_dir, limit=int(suspicious_limit))
    if suspicious_path is not None:
        summary.setdefault("artifacts", {})["suspicious_particles_csv"] = str(suspicious_path)
    return summary, residuals, suspicious_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize full COMSOL/reference diagnostics for an existing solver output.")
    parser.add_argument("--solver-output-dir", required=True, type=Path, help="Solver output directory.")
    parser.add_argument("--comsol-trajectory-csv", required=True, type=Path, help="COMSOL long trajectory/result CSV.")
    parser.add_argument("--comsol-release-csv", required=True, type=Path, help="COMSOL release/reference CSV for sampled particles.")
    parser.add_argument("--reference-scope", choices=("sampled", "full", "unspecified"), default="unspecified")
    parser.add_argument("--first-step-dir", type=Path, default=None, help="Optional first-step compare artifact directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("comsol_full_diagnostics"))
    parser.add_argument("--suspicious-limit", type=int, default=100, help="Maximum suspicious particle rows to write; use 0 to disable.")
    args = parser.parse_args(argv)

    summary, _, _ = build_full_diagnostics_summary(
        solver_output_dir=args.solver_output_dir,
        comsol_trajectory_csv=args.comsol_trajectory_csv,
        comsol_release_csv=args.comsol_release_csv,
        reference_scope=str(args.reference_scope),
        first_step_dir=args.first_step_dir,
        suspicious_limit=int(args.suspicious_limit),
        output_dir=args.output_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "full_comsol_diagnostics_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary_json": str(summary_path), **summary.get("artifacts", {})}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
