from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from ..core.boundary_service import (
    inside_geometry,
    nearest_boundary_edge_features_2d,
    sample_geometry_normal,
    sample_geometry_part_id,
    sample_geometry_sdf,
)
from ..io.runtime_builder import build_runtime_from_config


NEAR_WALL_NOHIT_COLUMNS = [
    "particle_id",
    "source_part_id",
    "source_provenance_group",
    "final_state_class",
    "x",
    "y",
    "z",
    "v_x",
    "v_y",
    "v_z",
    "nearest_boundary_part_id",
    "nearest_boundary_distance_m",
    "sdf_m",
    "inside_geometry",
    "normal_velocity_mps",
    "wall_hit_count",
    "wall_events_available",
    "release_grace_enabled",
    "release_grace_skip_count",
    "release_grace_blocked_count",
    "release_grace_blocked_reasons",
    "field_support_status",
    "classification",
    "classification_reason",
]


def _load_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return dict(payload)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _bool_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        return np.zeros(len(frame), dtype=bool)
    return frame[name].fillna(0).to_numpy(dtype=bool)


def _final_state_labels(final_particles: pd.DataFrame) -> np.ndarray:
    labels = np.full(len(final_particles), "inactive", dtype=object)
    labels[_bool_column(final_particles, "active")] = "active_free_flight"
    labels[_bool_column(final_particles, "contact_sliding")] = "contact_sliding"
    labels[_bool_column(final_particles, "contact_endpoint_stopped")] = "contact_endpoint_stopped"
    for name in ("invalid_mask_stopped", "numerical_boundary_stopped", "stuck", "absorbed", "escaped"):
        labels[_bool_column(final_particles, name)] = name
    return labels


def _spatial_dim(final_particles: pd.DataFrame, runtime: Any | None) -> int:
    if runtime is not None:
        return int(getattr(runtime, "spatial_dim", 2))
    return int(max(1, min(3, sum(1 for name in ("x", "y", "z") if name in final_particles.columns))))


def _value_or_nan(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name, np.nan)
    if value is None or (isinstance(value, str) and not value.strip()):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _int_or_default(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _position_array(final_particles: pd.DataFrame, spatial_dim: int) -> np.ndarray:
    data = []
    for axis in ("x", "y", "z")[: int(spatial_dim)]:
        if axis not in final_particles.columns:
            raise ValueError(f"final_particles.csv is missing position column {axis!r}")
        data.append(final_particles[axis].to_numpy(dtype=np.float64))
    return np.stack(data, axis=1)


def _velocity_array(final_particles: pd.DataFrame, spatial_dim: int) -> np.ndarray:
    data = []
    for axis in ("x", "y", "z")[: int(spatial_dim)]:
        underscored = f"v_{axis}"
        compact = f"v{axis}"
        if underscored in final_particles.columns:
            data.append(final_particles[underscored].to_numpy(dtype=np.float64))
        elif compact in final_particles.columns:
            data.append(final_particles[compact].to_numpy(dtype=np.float64))
        else:
            data.append(np.full(len(final_particles), np.nan, dtype=np.float64))
    return np.stack(data, axis=1)


def _normal_array(final_particles: pd.DataFrame, runtime: Any | None, positions: np.ndarray, spatial_dim: int) -> np.ndarray:
    data = []
    has_any = False
    for axis in ("x", "y", "z")[: int(spatial_dim)]:
        for name in (f"normal_{axis}", f"contact_normal_{axis}"):
            if name in final_particles.columns:
                data.append(final_particles[name].to_numpy(dtype=np.float64))
                has_any = True
                break
        else:
            data.append(np.full(len(final_particles), np.nan, dtype=np.float64))
    normals = np.stack(data, axis=1)
    if has_any:
        return normals
    if runtime is None:
        return normals
    sampled = []
    for position in positions:
        try:
            sampled.append(np.asarray(sample_geometry_normal(runtime, position), dtype=np.float64)[: int(spatial_dim)])
        except Exception:
            sampled.append(np.full(int(spatial_dim), np.nan, dtype=np.float64))
    return np.asarray(sampled, dtype=np.float64)


def _column_or_nan(final_particles: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    for name in names:
        if name in final_particles.columns:
            return final_particles[name].to_numpy(dtype=np.float64)
    return np.full(len(final_particles), np.nan, dtype=np.float64)


def _column_or_int_zero(final_particles: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    for name in names:
        if name in final_particles.columns:
            return final_particles[name].fillna(0).to_numpy(dtype=np.int64)
    return np.zeros(len(final_particles), dtype=np.int64)


def _inside_column(final_particles: pd.DataFrame, runtime: Any | None, positions: np.ndarray) -> np.ndarray:
    if "inside_geometry" in final_particles.columns:
        return final_particles["inside_geometry"].to_numpy(dtype=np.float64)
    if runtime is None:
        return np.full(len(final_particles), np.nan, dtype=np.float64)
    values = []
    for position in positions:
        try:
            values.append(float(bool(inside_geometry(runtime, position, on_boundary_tol_m=0.0))))
        except Exception:
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def _geometry_features(final_particles: pd.DataFrame, runtime: Any | None, spatial_dim: int) -> dict[str, np.ndarray]:
    positions = _position_array(final_particles, spatial_dim)
    velocities = _velocity_array(final_particles, spatial_dim)
    sdf = _column_or_nan(final_particles, ("sdf_m", "signed_sdf_m"))
    nearest_distance = _column_or_nan(final_particles, ("nearest_boundary_distance_m", "abs_sdf_m"))
    nearest_part = _column_or_int_zero(final_particles, ("nearest_boundary_part_id", "nearest_part_id", "part_id"))
    if runtime is not None:
        sampled_sdf = []
        sampled_part = []
        for position in positions:
            try:
                sampled_sdf.append(float(sample_geometry_sdf(runtime, position)))
            except Exception:
                sampled_sdf.append(float("nan"))
            try:
                sampled_part.append(int(sample_geometry_part_id(runtime, position)))
            except Exception:
                sampled_part.append(0)
        sampled_sdf_arr = np.asarray(sampled_sdf, dtype=np.float64)
        sdf = np.where(np.isfinite(sdf), sdf, sampled_sdf_arr)
        nearest_part = np.where(nearest_part != 0, nearest_part, np.asarray(sampled_part, dtype=np.int64))
        nearest_distance = np.where(np.isfinite(nearest_distance), nearest_distance, np.abs(sampled_sdf_arr))
        geometry_provider = getattr(runtime, "geometry_provider", None)
        geometry = getattr(geometry_provider, "geometry", None) if geometry_provider is not None else None
        if int(spatial_dim) == 2 and geometry is not None and getattr(geometry, "boundary_edges", None) is not None:
            edge_part, edge_distance = nearest_boundary_edge_features_2d(runtime, positions)
            finite = np.isfinite(edge_distance)
            nearest_part = np.where(finite, edge_part.astype(np.int64), nearest_part)
            nearest_distance = np.where(finite, edge_distance, nearest_distance)
    normals = _normal_array(final_particles, runtime, positions, spatial_dim)
    normal_velocity = np.full(len(final_particles), np.nan, dtype=np.float64)
    for idx in range(len(final_particles)):
        n = normals[idx]
        v = velocities[idx]
        if np.all(np.isfinite(n)) and np.all(np.isfinite(v)):
            mag = float(np.linalg.norm(n))
            if mag > 1.0e-30:
                normal_velocity[idx] = float(np.dot(v, n / mag))
    return {
        "positions": positions,
        "velocities": velocities,
        "sdf_m": sdf,
        "nearest_boundary_distance_m": nearest_distance,
        "nearest_boundary_part_id": nearest_part.astype(np.int64),
        "inside_geometry": _inside_column(final_particles, runtime, positions),
        "normal_velocity_mps": normal_velocity,
    }


def _wall_hit_counts(output_dir: Path) -> tuple[bool, dict[int, int]]:
    path = output_dir / "wall_events.csv"
    if not path.exists():
        return False, {}
    frame = pd.read_csv(path)
    if "particle_id" not in frame.columns or frame.empty:
        return True, {}
    counts = frame.groupby("particle_id").size()
    return True, {int(pid): int(count) for pid, count in counts.items()}


def _resolve_threshold(threshold_m: float | None, diagnostics: Mapping[str, Any], report: Mapping[str, Any]) -> float:
    if threshold_m is not None:
        value = float(threshold_m)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("--threshold-m must be a finite non-negative value")
        return float(value)
    candidates = []
    for source in (diagnostics, report):
        for key in ("on_boundary_tol_m", "epsilon_offset_m", "near_wall_threshold_m"):
            try:
                value = float(source.get(key, np.nan))
            except (TypeError, ValueError):
                value = float("nan")
            if np.isfinite(value) and value > 0.0:
                candidates.append(float(value))
    return float(max(candidates)) if candidates else 1.0e-6


def _field_support_status(row: Mapping[str, Any], diagnostics: Mapping[str, Any], report: Mapping[str, Any]) -> str:
    for name in ("field_support_status", "field_status", "valid_mask_status"):
        value = row.get(name, "")
        if value is not None and str(value).strip():
            return str(value)
    invalid_reason = str(row.get("invalid_stop_reason", "") or "").strip()
    if invalid_reason:
        return invalid_reason
    if int(diagnostics.get("valid_mask_hard_invalid_count", report.get("valid_mask_hard_invalid_count", 0)) or 0) > 0:
        return "global_hard_invalid_seen"
    if int(diagnostics.get("valid_mask_mixed_stencil_count", report.get("valid_mask_mixed_stencil_count", 0)) or 0) > 0:
        return "global_mixed_stencil_seen"
    return "unknown"


def _is_field_support_issue(status: str) -> bool:
    text = str(status).strip().lower()
    return any(token in text for token in ("field", "valid_mask", "hard_invalid", "mixed_stencil", "support"))


def _classify_row(
    *,
    row: Mapping[str, Any],
    nearest_part_id: int,
    wall_events_available: bool,
    wall_hit_count: int,
    diagnostics: Mapping[str, Any],
    report: Mapping[str, Any],
    field_support_status: str,
) -> tuple[str, str]:
    source_part_id = _int_or_default(row.get("source_part_id", 0), 0)
    same_source_hint = source_part_id > 0 and int(nearest_part_id) == int(source_part_id)
    release_skip = int(diagnostics.get("source_surface_release_skip_count", report.get("source_surface_release_skip_count", 0)) or 0)
    release_blocked = int(
        diagnostics.get(
            "source_surface_release_skip_blocked_count",
            report.get("source_surface_release_skip_blocked_count", 0),
        )
        or 0
    )
    if not wall_events_available:
        return "no_wall_events_available", "wall_events.csv was not present, so per-particle hit history is unavailable"
    if _is_field_support_issue(field_support_status):
        return "field_support_issue", f"field support status is {field_support_status!r}"
    if source_part_id <= 0:
        return "unknown_source_provenance", "source_part_id <= 0 cannot establish same-source release provenance"
    if release_skip > 0 and same_source_hint:
        return "crossing_skipped_by_release_grace", "global release-grace skip count is nonzero and nearest part matches source"
    if release_blocked > 0 and same_source_hint:
        return "release_grace_blocked", "global release-grace blocked count is nonzero and nearest part matches source"
    if bool(row.get("numerical_boundary_stopped", 0)) or int(diagnostics.get("unresolved_crossing_count", 0) or 0) > 0:
        return "unresolved_crossing_numerical_boundary_issue", "numerical boundary or unresolved crossing diagnostics are present"
    if wall_hit_count <= 0:
        return "no_segment_crossing_recorded", "particle is active near a boundary but has no recorded wall hit"
    return "unresolved_crossing_numerical_boundary_issue", "particle remains active near a boundary despite recorded wall interactions"


def _top_counts(values: Sequence[Any], key_name: str) -> list[dict[str, Any]]:
    counts = pd.Series(list(values), dtype=object).value_counts(dropna=False)
    return [{key_name: str(key), "count": int(value)} for key, value in counts.head(12).items()]


def _load_runtime(config_path: Path | None) -> Any | None:
    if config_path is None:
        return None
    path = Path(config_path).resolve()
    return build_runtime_from_config(_load_yaml_mapping(path), path.parent)


def analyze_near_wall_nohit(
    *,
    output_dir: Path,
    config_path: Path | None = None,
    threshold_m: float | None = None,
    analysis_output_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    analysis_dir = output_dir if analysis_output_dir is None else Path(analysis_output_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "final_particles.csv"
    if not final_path.exists():
        raise FileNotFoundError(final_path)
    final_particles = pd.read_csv(final_path)
    if "particle_id" not in final_particles.columns:
        raise ValueError("final_particles.csv is missing required column 'particle_id'")

    diagnostics = _load_json_optional(output_dir / "collision_diagnostics.json")
    report = _load_json_optional(output_dir / "solver_report.json")
    runtime = _load_runtime(config_path)
    spatial_dim = _spatial_dim(final_particles, runtime)
    threshold = _resolve_threshold(threshold_m, diagnostics, report)
    features = _geometry_features(final_particles, runtime, spatial_dim)
    states = _final_state_labels(final_particles)
    wall_events_available, wall_counts = _wall_hit_counts(output_dir)

    active_mask = _bool_column(final_particles, "active")
    if "active" not in final_particles.columns:
        active_mask = np.isin(states, ("active_free_flight", "contact_sliding"))
    distance = np.asarray(features["nearest_boundary_distance_m"], dtype=np.float64)
    sdf = np.asarray(features["sdf_m"], dtype=np.float64)
    near_mask = (np.isfinite(distance) & (distance <= float(threshold))) | (
        np.isfinite(sdf) & (np.abs(sdf) <= float(threshold))
    )

    rows: list[dict[str, Any]] = []
    for idx, raw_row in final_particles.iterrows():
        particle_id = _int_or_default(raw_row.get("particle_id", 0), 0)
        wall_hit_count = int(wall_counts.get(int(particle_id), 0))
        no_recorded_hit = (not wall_events_available) or wall_hit_count <= 0
        if not bool(active_mask[idx]) or not bool(near_mask[idx]) or not bool(no_recorded_hit):
            continue
        field_status = _field_support_status(raw_row, diagnostics, report)
        nearest_part_id = int(features["nearest_boundary_part_id"][idx])
        classification, reason = _classify_row(
            row=raw_row,
            nearest_part_id=nearest_part_id,
            wall_events_available=bool(wall_events_available),
            wall_hit_count=int(wall_hit_count),
            diagnostics=diagnostics,
            report=report,
            field_support_status=field_status,
        )
        positions = features["positions"][idx]
        velocities = features["velocities"][idx]
        inside_value = float(features["inside_geometry"][idx])
        item: dict[str, Any] = {
            "particle_id": int(particle_id),
            "source_part_id": _int_or_default(raw_row.get("source_part_id", 0), 0),
            "source_provenance_group": str(raw_row.get("source_provenance_group", "")),
            "final_state_class": str(states[idx]),
            "x": float(positions[0]) if spatial_dim >= 1 else float("nan"),
            "y": float(positions[1]) if spatial_dim >= 2 else float("nan"),
            "z": float(positions[2]) if spatial_dim >= 3 else float("nan"),
            "v_x": float(velocities[0]) if spatial_dim >= 1 else float("nan"),
            "v_y": float(velocities[1]) if spatial_dim >= 2 else float("nan"),
            "v_z": float(velocities[2]) if spatial_dim >= 3 else float("nan"),
            "nearest_boundary_part_id": int(nearest_part_id),
            "nearest_boundary_distance_m": float(distance[idx]),
            "sdf_m": float(sdf[idx]),
            "inside_geometry": int(inside_value) if np.isfinite(inside_value) else "",
            "normal_velocity_mps": float(features["normal_velocity_mps"][idx]),
            "wall_hit_count": int(wall_hit_count),
            "wall_events_available": int(bool(wall_events_available)),
            "release_grace_enabled": int(
                diagnostics.get(
                    "source_surface_release_grace_enabled",
                    report.get("source_surface_release_grace_enabled", 0),
                )
                or 0
            ),
            "release_grace_skip_count": int(
                diagnostics.get("source_surface_release_skip_count", report.get("source_surface_release_skip_count", 0))
                or 0
            ),
            "release_grace_blocked_count": int(
                diagnostics.get(
                    "source_surface_release_skip_blocked_count",
                    report.get("source_surface_release_skip_blocked_count", 0),
                )
                or 0
            ),
            "release_grace_blocked_reasons": json.dumps(
                diagnostics.get(
                    "source_surface_release_skip_blocked_reasons",
                    report.get("source_surface_release_skip_blocked_reasons", {}),
                ),
                sort_keys=True,
            ),
            "field_support_status": str(field_status),
            "classification": str(classification),
            "classification_reason": str(reason),
        }
        rows.append(item)

    particles_path = analysis_dir / "near_wall_nohit_particles.csv"
    pd.DataFrame(rows, columns=NEAR_WALL_NOHIT_COLUMNS).to_csv(particles_path, index=False)
    classification_counts = (
        pd.Series([row["classification"] for row in rows], dtype=object).value_counts().to_dict()
        if rows
        else {}
    )
    summary = {
        "output_dir": str(output_dir),
        "config": None if config_path is None else str(Path(config_path)),
        "threshold_m": float(threshold),
        "geometry_available": int(runtime is not None or "nearest_boundary_distance_m" in final_particles.columns or "sdf_m" in final_particles.columns),
        "wall_events_available": int(bool(wall_events_available)),
        "collision_diagnostics_available": int((output_dir / "collision_diagnostics.json").exists()),
        "final_particle_count": int(len(final_particles)),
        "active_particle_count": int(np.count_nonzero(active_mask)),
        "near_wall_active_count": int(np.count_nonzero(active_mask & near_mask)),
        "suspicious_particle_count": int(len(rows)),
        "classification_counts": {str(key): int(value) for key, value in classification_counts.items()},
        "nearest_boundary_part_counts": _top_counts([row["nearest_boundary_part_id"] for row in rows], "part_id"),
        "artifacts": {
            "near_wall_nohit_particles_csv": str(particles_path),
            "near_wall_nohit_summary_json": str(analysis_dir / "near_wall_nohit_summary.json"),
        },
        "interpretation": (
            "Rows are active particles near a boundary with no recorded wall hit. Use classification together "
            "with source_part_id, nearest part, SDF/inside status, normal velocity, release-grace counters, and "
            "field support status to decide whether the next fix belongs in release semantics, segment crossing, "
            "field support, source provenance, or integrator/force behavior."
        ),
    }
    summary_path = analysis_dir / "near_wall_nohit_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze active near-wall final particles that have no recorded wall hit."
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Existing solver output directory")
    parser.add_argument("--config", type=Path, default=None, help="Optional run_config.yaml used to sample geometry")
    parser.add_argument(
        "--threshold-m",
        type=float,
        default=None,
        help="Near-wall distance threshold. Defaults to diagnostics tolerances when available, otherwise 1e-6.",
    )
    parser.add_argument(
        "--analysis-output-dir",
        type=Path,
        default=None,
        help="Optional directory for the two analysis artifacts. Defaults to --output-dir.",
    )
    args = parser.parse_args(argv)
    analyze_near_wall_nohit(
        output_dir=args.output_dir,
        config_path=args.config,
        threshold_m=args.threshold_m,
        analysis_output_dir=args.analysis_output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
