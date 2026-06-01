from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


FORCE_COMPONENTS = (
    "drag",
    "electric",
    "thermo",
    "dielectrophoretic",
    "lift",
    "pressure_gradient",
    "virtual_mass",
    "brownian",
    "external",
    "total",
)
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
        return {"count": 0, "mean": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
        "p95": float(np.percentile(finite, 95.0)),
    }


def _load_json(path: Path | None) -> dict[str, Any] | None:
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


def _has_tmp_component(path: Path | None) -> bool:
    if path is None:
        return False
    return any(str(part).lower().startswith("_tmp") for part in path.parts)


def _validate_input_paths(paths: Sequence[Path | None], *, allow_tmp_inputs: bool) -> None:
    if bool(allow_tmp_inputs):
        return
    bad = [str(path) for path in paths if _has_tmp_component(path)]
    if bad:
        raise ValueError(
            "Refusing _tmp* input paths as official residual-gap sources. "
            "Move/copy current artifacts to a named output root or pass --allow-tmp-inputs for ad hoc inspection: "
            + ", ".join(bad)
        )


def _status_counts(report: Mapping[str, Any] | None) -> dict[str, int]:
    counts = report.get("status_counts", {}) if isinstance(report, Mapping) else {}
    if not isinstance(counts, Mapping):
        return {}
    return {str(k): int(v) for k, v in counts.items() if isinstance(v, (int, float))}


def _summarize_import_preflight(
    provider_report: Mapping[str, Any] | None,
    input_report: Mapping[str, Any] | None,
    prepared_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "provider": {
            "available": int(provider_report is not None),
            "passed": None if provider_report is None else bool(provider_report.get("passed", False)),
            "coordinate_system": "" if provider_report is None else str(provider_report.get("coordinate_system", "")),
            "field_backend_kind": "" if provider_report is None else str(provider_report.get("field_backend_kind", "")),
            "status_counts": _status_counts(provider_report),
        },
        "input": {
            "available": int(input_report is not None),
            "passed": None if input_report is None else bool(input_report.get("passed", False)),
            "mode": "" if input_report is None else str(input_report.get("mode", "")),
            "status_counts": _status_counts(input_report),
        },
        "prepared": {
            "available": int(prepared_summary is not None),
            "coordinate_system": "" if prepared_summary is None else str(prepared_summary.get("coordinate_system", "")),
            "spatial_dim": None if prepared_summary is None else prepared_summary.get("spatial_dim"),
            "axis_names": [] if prepared_summary is None else list(prepared_summary.get("axis_names", [])),
        },
    }


def _summarize_release(
    prepared_summary: Mapping[str, Any] | None,
    source_model_summary: Mapping[str, Any] | None,
    solver_report: Mapping[str, Any] | None,
    source_diag: pd.DataFrame | None,
) -> dict[str, Any]:
    source_summary = {}
    if isinstance(prepared_summary, Mapping) and isinstance(prepared_summary.get("source_model_summary"), Mapping):
        source_summary.update(dict(prepared_summary["source_model_summary"]))
    if isinstance(source_model_summary, Mapping):
        source_summary.update(dict(source_model_summary))
    particle_count = None
    if source_diag is not None:
        particle_count = int(len(source_diag))
    elif solver_report is not None:
        particle_count = int(solver_report.get("particle_count", 0))
    return {
        "metric_scope": "post_preprocess_when_source_diagnostics_available_else_solver_final_counts",
        "particle_count": particle_count,
        "released_count_final": None if solver_report is None else int(solver_report.get("released_count", 0)),
        "boundary_release_enabled": int(source_summary.get("boundary_release_enabled", 0) or 0),
        "boundary_release_applied_count": int(source_summary.get("boundary_release_applied_count", 0) or 0),
        "boundary_release_failed_offset_count": int(source_summary.get("boundary_release_failed_offset_count", 0) or 0),
        "boundary_release_capture_tolerance_m": source_summary.get("boundary_release_capture_tolerance_m"),
        "boundary_release_inward_offset_m": source_summary.get("boundary_release_inward_offset_m"),
        "source_provenance_counts": dict(source_summary.get("source_provenance_counts", {}))
        if isinstance(source_summary.get("source_provenance_counts", {}), Mapping)
        else {},
    }


def _summarize_field_support(
    provider_report: Mapping[str, Any] | None,
    input_report: Mapping[str, Any] | None,
    solver_report: Mapping[str, Any] | None,
    collision_diag: Mapping[str, Any] | None,
) -> dict[str, Any]:
    runtime_source = collision_diag if collision_diag is not None else solver_report
    runtime_source = runtime_source if isinstance(runtime_source, Mapping) else {}
    return {
        "preflight_provider_status_counts": _status_counts(provider_report),
        "preflight_input_status_counts": _status_counts(input_report),
        "runtime_mixed_stencil_count": int(runtime_source.get("valid_mask_mixed_stencil_count", 0) or 0),
        "runtime_hard_invalid_count": int(runtime_source.get("valid_mask_hard_invalid_count", 0) or 0),
        "runtime_invalid_mask_stopped_count": int(runtime_source.get("invalid_mask_stopped_count", 0) or 0),
    }


def _summarize_first_step(first_step_summary: Mapping[str, Any] | None, first_step_error: pd.DataFrame | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": int(first_step_summary is not None or first_step_error is not None),
        "metric_scope": "post_preprocess_to_post_first_step",
    }
    if isinstance(first_step_summary, Mapping):
        out.update(
            {
                "particle_count": first_step_summary.get("particle_count"),
                "compared_particle_count": first_step_summary.get("compared_particle_count"),
                "stochastic_policy": first_step_summary.get("stochastic_policy"),
                "position_error_m": first_step_summary.get("position_error_m", {}),
                "velocity_error_mps": first_step_summary.get("velocity_error_mps", {}),
            }
        )
    if first_step_error is not None and not first_step_error.empty:
        for column in ("position_error_m", "velocity_error_mps", "speed_ratio"):
            if column in first_step_error.columns:
                out[column] = _finite_summary(pd.to_numeric(first_step_error[column], errors="coerce").to_numpy())
    return out


def _summarize_force_contributions(first_step_summary: Mapping[str, Any] | None, force_df: pd.DataFrame | None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "available": int(force_df is not None or first_step_summary is not None),
        "metric_scope": "post_preprocess_state_deterministic_components",
        "enabled_forces": list(first_step_summary.get("enabled_forces", [])) if isinstance(first_step_summary, Mapping) else [],
        "rows": 0 if force_df is None else int(len(force_df)),
        "component_acceleration_norm_mps2": {},
    }
    if force_df is None or force_df.empty:
        return out
    for component in FORCE_COMPONENTS:
        cols = [col for col in force_df.columns if str(col).startswith(f"{component}_a")]
        if not cols:
            continue
        values = force_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        norm = np.linalg.norm(values, axis=1)
        out["component_acceleration_norm_mps2"][component] = _finite_summary(norm)
    return out


def _state_counts_from_final(final_df: pd.DataFrame | None) -> dict[str, int]:
    if final_df is None:
        return {}
    keys = ("active", "stuck", "absorbed", "escaped", "invalid_mask_stopped", "numerical_boundary_stopped")
    counts: dict[str, int] = {}
    for key in keys:
        if key in final_df.columns:
            counts[key] = int(pd.to_numeric(final_df[key], errors="coerce").fillna(0).sum())
    return counts


def _summarize_final_state(solver_report: Mapping[str, Any] | None, final_df: pd.DataFrame | None) -> dict[str, Any]:
    count = int(solver_report.get("particle_count", 0)) if isinstance(solver_report, Mapping) else 0
    if final_df is not None and count <= 0:
        count = int(len(final_df))
    counts = _state_counts_from_final(final_df)
    if isinstance(solver_report, Mapping):
        for key in ("released", "stuck", "absorbed", "escaped", "invalid_mask_stopped", "numerical_boundary_stopped"):
            report_key = f"{key}_count"
            if report_key in solver_report:
                counts[key] = int(solver_report.get(report_key, 0) or 0)
    fractions = {key: (float(value) / float(count) if count > 0 else None) for key, value in counts.items()}
    provenance_counts: dict[str, int] = {}
    if final_df is not None and "source_provenance_group" in final_df.columns:
        provenance_counts = {
            str(key): int(value)
            for key, value in final_df["source_provenance_group"].fillna("unknown_source").astype(str).value_counts().items()
        }
    return {
        "metric_scope": "final_snapshot",
        "particle_count": int(count),
        "counts": counts,
        "fractions": fractions,
        "source_provenance_counts": provenance_counts,
    }


def _summarize_wall_interactions(
    solver_report: Mapping[str, Any] | None,
    collision_diag: Mapping[str, Any] | None,
    wall_summary: Mapping[str, Any] | None,
    boundary_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    source = collision_diag if collision_diag is not None else solver_report
    source = source if isinstance(source, Mapping) else {}
    return {
        "available": int(bool(source) or wall_summary is not None or boundary_summary is not None),
        "metric_scope": "ever_reached_wall_event_artifacts_when_available",
        "wall_summary": wall_summary or {},
        "boundary_first_hit_compare": boundary_summary or {},
        "primary_hit_count": int(source.get("primary_hit_count", 0) or 0),
        "wall_events_written": None if solver_report is None else int(solver_report.get("wall_events_written", 0) or 0),
        "unresolved_crossing_count": int(source.get("unresolved_crossing_count", 0) or 0),
        "max_hits_reached_count": int(source.get("max_hits_reached_count", 0) or 0),
        "nearest_projection_fallback_count": int(source.get("nearest_projection_fallback_count", 0) or 0),
    }


def _summarize_near_wall_active_no_hit(
    solver_report: Mapping[str, Any] | None,
    collision_diag: Mapping[str, Any] | None,
    near_wall_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(near_wall_summary, Mapping):
        return {
            "available": 1,
            "metric_scope": "requested_near_wall_nohit_analysis",
            "summary": dict(near_wall_summary),
        }
    source = collision_diag if collision_diag is not None else solver_report
    state_summary = source.get("state_geometry_summary", {}) if isinstance(source, Mapping) else {}
    by_state = state_summary.get("by_state", {}) if isinstance(state_summary, Mapping) else {}
    active = {}
    for key in ("active_free_flight", "active"):
        if isinstance(by_state, Mapping) and isinstance(by_state.get(key), Mapping):
            active = dict(by_state[key])
            break
    return {
        "available": int(bool(active)),
        "metric_scope": "final_snapshot_active_near_boundary_proxy_not_ever_reached",
        "near_boundary_threshold_m": state_summary.get("near_boundary_threshold_m") if isinstance(state_summary, Mapping) else None,
        "active_count": int(active.get("count", 0) or 0) if active else None,
        "active_near_boundary_count": int(active.get("near_boundary_count", 0) or 0) if active else None,
        "active_nearest_part_counts": active.get("nearest_part_counts", []) if active else [],
    }


def _load_first_crossing_or_vacuum(first_crossing_path: Path | None, vacuum_time_path: Path | None) -> dict[str, Any]:
    first_crossing = _load_json(first_crossing_path) if first_crossing_path and first_crossing_path.suffix.lower() == ".json" else None
    vacuum = _load_json(vacuum_time_path) if vacuum_time_path and vacuum_time_path.suffix.lower() == ".json" else None
    if vacuum is None and vacuum_time_path is not None and vacuum_time_path.exists() and vacuum_time_path.suffix.lower() == ".csv":
        frame = pd.read_csv(vacuum_time_path)
        vacuum = {"row_count": int(len(frame))}
        for col in frame.columns:
            if "time" in str(col).lower():
                vacuum[str(col)] = _finite_summary(pd.to_numeric(frame[col], errors="coerce").to_numpy())
    return {
        "available": int(first_crossing is not None or vacuum is not None),
        "metric_scope": "ever_reached_trajectory_metric_when_artifact_available",
        "first_crossing": first_crossing or {},
        "vacuum_time": vacuum or {},
    }


def _summarize_runtime_counters(solver_report: Mapping[str, Any] | None, collision_diag: Mapping[str, Any] | None) -> dict[str, Any]:
    source = {}
    if isinstance(solver_report, Mapping):
        source.update(dict(solver_report))
    if isinstance(collision_diag, Mapping):
        source.update(dict(collision_diag))
    return {
        "timing_s": dict(solver_report.get("timing_s", {})) if isinstance(solver_report, Mapping) and isinstance(solver_report.get("timing_s"), Mapping) else {},
        "counters": {key: int(source.get(key, 0) or 0) for key in COUNTER_KEYS},
    }


def _summarize_ensemble(ensemble_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if ensemble_summary is None:
        return {"available": 0}
    runs = []
    for row in ensemble_summary.get("runs", []) if isinstance(ensemble_summary.get("runs", []), list) else []:
        if not isinstance(row, Mapping):
            continue
        runs.append(
            {
                "run": row.get("run", ""),
                "particle_count": row.get("particle_count"),
                "class_match_ratio_vs_reference": row.get("class_match_ratio_vs_reference"),
                "class_mismatch_count_vs_reference": row.get("class_mismatch_count_vs_reference"),
                "unresolved_crossing_count": row.get("unresolved_crossing_count"),
                "boundary_event_failure_count": row.get("boundary_event_failure_count"),
                "geometry_feature_delta_vs_reference": row.get("geometry_feature_delta_vs_reference", {}),
            }
        )
    return {
        "available": 1,
        "metric_scope": "ensemble_compare_summary_uses_final_snapshot_state_classes_and_geometry_features",
        "reference_scope": ensemble_summary.get("reference_scope", "unspecified"),
        "comparison_dir": ensemble_summary.get("comparison_dir", ""),
        "runs": runs,
        "pair_delta": ensemble_summary.get("pair_delta", {}),
    }


def _summarize_full_comsol_diagnostics(full_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if full_summary is None:
        return {"available": 0}
    return {
        "available": 1,
        "reference_scope": full_summary.get("reference_scope", "unspecified"),
        "comsol_reference_counts": full_summary.get("comsol_reference_counts", {}),
        "final_state": full_summary.get("final_state", {}),
        "preprocess": full_summary.get("preprocess", {}),
        "wall_events": full_summary.get("wall_events", {}),
        "near_wall_active": full_summary.get("near_wall_active", {}),
        "top_source_parts_for_residuals": full_summary.get("top_source_parts_for_residuals", []),
    }


def build_summary(
    *,
    run_output_dir: Path | None,
    preflight_dir: Path | None,
    first_step_dir: Path | None,
    boundary_summary_path: Path | None,
    ensemble_summary_path: Path | None,
    near_wall_summary_path: Path | None,
    full_comsol_diagnostics_path: Path | None,
    first_crossing_path: Path | None,
    vacuum_time_path: Path | None,
    reference_scope: str,
) -> dict[str, Any]:
    preflight_root = preflight_dir or run_output_dir
    first_step_root = first_step_dir
    provider_report_path = _discover(preflight_root, "provider_contract_report.json")
    input_report_path = _discover(preflight_root, "input_contract_report.json")
    prepared_summary_path = _discover(run_output_dir, "prepared_runtime_summary.json") or _discover(preflight_root, "prepared_runtime_summary.json")
    source_model_summary_path = _discover(preflight_root, "source_model_summary.json")
    source_diag_path = _discover(preflight_root, "source_particle_diagnostics.csv")
    solver_report_path = _discover(run_output_dir, "solver_report.json")
    collision_diag_path = _discover(run_output_dir, "collision_diagnostics.json")
    final_particles_path = _discover(run_output_dir, "final_particles.csv")
    wall_summary_path = _discover(run_output_dir, "wall_summary.json")
    near_wall_summary_path = near_wall_summary_path or _discover(run_output_dir, "near_wall_nohit_summary.json")
    first_step_summary_path = _discover(first_step_root, "first_step_compare_summary.json")
    first_step_error_path = _discover(first_step_root, "first_step_error.csv")
    force_contributions_path = _discover(first_step_root, "force_contributions.csv")
    full_comsol_diagnostics_path = (
        full_comsol_diagnostics_path
        or _discover(run_output_dir, "full_comsol_diagnostics_summary.json")
    )
    first_crossing_path = first_crossing_path or _discover(run_output_dir, "first_crossing_summary.json")
    vacuum_time_path = vacuum_time_path or _discover(run_output_dir, "vacuum_time_summary.json") or _discover(run_output_dir, "vacuum_time_summary.csv")

    artifacts = [
        _artifact("provider_contract_report", provider_report_path),
        _artifact("input_contract_report", input_report_path),
        _artifact("prepared_runtime_summary", prepared_summary_path),
        _artifact("source_model_summary", source_model_summary_path),
        _artifact("source_particle_diagnostics", source_diag_path),
        _artifact("solver_report", solver_report_path, required=run_output_dir is not None),
        _artifact("collision_diagnostics", collision_diag_path),
        _artifact("final_particles", final_particles_path, required=run_output_dir is not None),
        _artifact("wall_summary", wall_summary_path),
        _artifact("first_step_compare_summary", first_step_summary_path),
        _artifact("first_step_error", first_step_error_path),
        _artifact("force_contributions", force_contributions_path),
        _artifact("boundary_hit_comparison_summary", boundary_summary_path),
        _artifact("ensemble_comparison_summary", ensemble_summary_path),
        _artifact("near_wall_nohit_summary", near_wall_summary_path),
        _artifact("full_comsol_diagnostics_summary", full_comsol_diagnostics_path),
        _artifact("first_crossing_summary", first_crossing_path),
        _artifact("vacuum_time_summary", vacuum_time_path),
    ]

    provider_report = _load_json(provider_report_path)
    input_report = _load_json(input_report_path)
    prepared_summary = _load_json(prepared_summary_path)
    source_model_summary = _load_json(source_model_summary_path)
    solver_report = _load_json(solver_report_path)
    collision_diag = _load_json(collision_diag_path)
    wall_summary = _load_json(wall_summary_path)
    first_step_summary = _load_json(first_step_summary_path)
    boundary_summary = _load_json(boundary_summary_path)
    ensemble_summary = _load_json(ensemble_summary_path)
    near_wall_summary = _load_json(near_wall_summary_path)
    full_comsol_diagnostics = _load_json(full_comsol_diagnostics_path)
    final_particles = _read_csv(final_particles_path)
    source_diag = _read_csv(source_diag_path)
    first_step_error = _read_csv(first_step_error_path)
    force_df = _read_csv(force_contributions_path)

    effective_scope = str(reference_scope)
    if effective_scope == "unspecified" and isinstance(ensemble_summary, Mapping):
        effective_scope = str(ensemble_summary.get("reference_scope", "unspecified"))

    return {
        "summary_schema_version": 1,
        "reference_scope": effective_scope,
        "metric_scope_notes": [
            "final_state metrics are final snapshot metrics from final_particles.csv/solver_report.json",
            "wall interaction, first-crossing, and vacuum-time metrics are ever-reached metrics only when their artifacts are available",
            "sampled and full COMSOL/reference scopes are operator-declared; this tool does not infer scope from paths",
        ],
        "artifact_status": artifacts,
        "missing_optional_artifacts": [item["name"] for item in artifacts if item["status"] == "missing_optional"],
        "missing_required_artifacts": [item["name"] for item in artifacts if item["status"] == "missing_required"],
        "import_preflight": _summarize_import_preflight(provider_report, input_report, prepared_summary),
        "release": _summarize_release(prepared_summary, source_model_summary, solver_report, source_diag),
        "field_support": _summarize_field_support(provider_report, input_report, solver_report, collision_diag),
        "first_step": _summarize_first_step(first_step_summary, first_step_error),
        "force_contributions": _summarize_force_contributions(first_step_summary, force_df),
        "wall_interactions": _summarize_wall_interactions(solver_report, collision_diag, wall_summary, boundary_summary),
        "near_wall_active_no_hit": _summarize_near_wall_active_no_hit(solver_report, collision_diag, near_wall_summary),
        "final_state": _summarize_final_state(solver_report, final_particles),
        "first_crossing_vacuum_time": _load_first_crossing_or_vacuum(first_crossing_path, vacuum_time_path),
        "runtime_collision_counters": _summarize_runtime_counters(solver_report, collision_diag),
        "ensemble": _summarize_ensemble(ensemble_summary),
        "full_comsol_diagnostics": _summarize_full_comsol_diagnostics(full_comsol_diagnostics),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return f"`{json.dumps(_json_safe(value), sort_keys=True)}`"
    return str(value)


def write_markdown_report(summary: Mapping[str, Any], path: Path) -> None:
    final_state = summary.get("final_state", {}) if isinstance(summary.get("final_state"), Mapping) else {}
    first_step = summary.get("first_step", {}) if isinstance(summary.get("first_step"), Mapping) else {}
    wall = summary.get("wall_interactions", {}) if isinstance(summary.get("wall_interactions"), Mapping) else {}
    counters = summary.get("runtime_collision_counters", {}) if isinstance(summary.get("runtime_collision_counters"), Mapping) else {}
    counter_values = counters.get("counters", {}) if isinstance(counters, Mapping) and isinstance(counters.get("counters"), Mapping) else {}
    lines = [
        "# Current COMSOL Residual Gap Summary",
        "",
        f"- Reference scope: `{summary.get('reference_scope', 'unspecified')}`",
        f"- Missing required artifacts: {_fmt(summary.get('missing_required_artifacts', []))}",
        f"- Missing optional artifacts: {_fmt(summary.get('missing_optional_artifacts', []))}",
        "",
        "## Import And Preflight",
        "",
        f"- Provider: {_fmt(summary.get('import_preflight', {}).get('provider', {}))}",
        f"- Input: {_fmt(summary.get('import_preflight', {}).get('input', {}))}",
        "",
        "## Release And Field Support",
        "",
        f"- Release: {_fmt(summary.get('release', {}))}",
        f"- Field support: {_fmt(summary.get('field_support', {}))}",
        "",
        "## First Step And Forces",
        "",
        f"- First-step position error: {_fmt(first_step.get('position_error_m'))}",
        f"- First-step velocity error: {_fmt(first_step.get('velocity_error_mps'))}",
        f"- First-step speed ratio: {_fmt(first_step.get('speed_ratio'))}",
        f"- Force contributions: {_fmt(summary.get('force_contributions', {}))}",
        "",
        "## Walls And Runtime",
        "",
        f"- Wall interactions: {_fmt(wall)}",
        f"- Near-wall active no-hit proxy: {_fmt(summary.get('near_wall_active_no_hit', {}))}",
        f"- First-crossing / vacuum-time: {_fmt(summary.get('first_crossing_vacuum_time', {}))}",
        f"- Runtime counters: {_fmt(counter_values)}",
        "",
        "## Final Snapshot And Ensemble",
        "",
        f"- Final state: {_fmt(final_state)}",
        f"- Ensemble: {_fmt(summary.get('ensemble', {}))}",
        "",
        "Final snapshot metrics and ever-reached trajectory metrics are intentionally reported separately.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize current residual gaps against COMSOL/reference artifacts.")
    parser.add_argument("--run-output-dir", type=Path, default=None, help="Solver output directory for the candidate run.")
    parser.add_argument("--preflight-dir", type=Path, default=None, help="Optional --check-input output directory.")
    parser.add_argument("--first-step-dir", type=Path, default=None, help="Directory from particle_tracer_unified.compare.first_step_compare.")
    parser.add_argument("--boundary-summary", type=Path, default=None, help="boundary_hit_comparison.json from boundary_compare.")
    parser.add_argument("--ensemble-summary", type=Path, default=None, help="comparison_summary.json from compare_against_reference.")
    parser.add_argument("--near-wall-nohit-summary", type=Path, default=None, help="Optional near_wall_nohit_summary.json.")
    parser.add_argument("--comsol-full-diagnostics-summary", type=Path, default=None, help="Optional full_comsol_diagnostics_summary.json.")
    parser.add_argument("--first-crossing-summary", type=Path, default=None, help="Optional first-crossing summary JSON.")
    parser.add_argument("--vacuum-time-summary", type=Path, default=None, help="Optional vacuum-time summary JSON/CSV.")
    parser.add_argument("--reference-scope", choices=("sampled", "full", "unspecified"), default="unspecified")
    parser.add_argument("--output-dir", type=Path, default=Path("comsol_residual_gap"))
    parser.add_argument("--allow-tmp-inputs", action="store_true", help="Allow _tmp* input paths for ad hoc inspection.")
    args = parser.parse_args(argv)

    input_paths = [
        args.run_output_dir,
        args.preflight_dir,
        args.first_step_dir,
        args.boundary_summary,
        args.ensemble_summary,
        args.near_wall_nohit_summary,
        args.comsol_full_diagnostics_summary,
        args.first_crossing_summary,
        args.vacuum_time_summary,
    ]
    _validate_input_paths(input_paths, allow_tmp_inputs=bool(args.allow_tmp_inputs))
    summary = build_summary(
        run_output_dir=args.run_output_dir,
        preflight_dir=args.preflight_dir,
        first_step_dir=args.first_step_dir,
        boundary_summary_path=args.boundary_summary,
        ensemble_summary_path=args.ensemble_summary,
        near_wall_summary_path=args.near_wall_nohit_summary,
        full_comsol_diagnostics_path=args.comsol_full_diagnostics_summary,
        first_crossing_path=args.first_crossing_summary,
        vacuum_time_path=args.vacuum_time_summary,
        reference_scope=str(args.reference_scope),
    )

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "current_residual_gap_summary.json"
    report_path = out / "current_residual_gap_report.md"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    write_markdown_report(summary, report_path)
    print(json.dumps({"summary_json": str(summary_path), "report_md": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
