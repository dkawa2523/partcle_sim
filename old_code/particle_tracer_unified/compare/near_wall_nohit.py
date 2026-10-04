from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from particle_tracer_unified.application import load_case
from particle_tracer_unified.compare._common import json_safe
from particle_tracer_unified.core.boundary_service import (
    inside_geometry,
    nearest_boundary_edge_features_2d,
    sample_geometry_normal,
    sample_geometry_part_id,
    sample_geometry_sdf,
)

NEAR_WALL_NOHIT_COLUMNS = [
    "particle_id",
    "source_part_id",
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
    "field_support_status",
    "classification",
    "classification_reason",
]


def _load_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _final_state_labels(final_particles: pd.DataFrame) -> np.ndarray:
    if "final_state" not in final_particles.columns:
        raise ValueError("final_particles.csv is missing required column 'final_state'")
    return (
        final_particles["final_state"]
        .fillna("inactive")
        .astype(str)
        .to_numpy(dtype=object)
    )


def _spatial_dim(final_particles: pd.DataFrame, runtime: Any | None) -> int:
    if runtime is not None:
        return int(getattr(runtime, "spatial_dim", 2))
    cartesian = sum(
        1 for name in ("x_m", "y_m", "z_m") if name in final_particles.columns
    )
    axisymmetric = sum(1 for name in ("r_m", "z_m") if name in final_particles.columns)
    return int(max(1, min(3, max(cartesian, axisymmetric))))


def _int_or_default(value: Any, default: int = 0) -> int:
    try:
        if value is None or (isinstance(value, str) and not value.strip()):
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _position_array(final_particles: pd.DataFrame, spatial_dim: int) -> np.ndarray:
    data = []
    axes = (
        ("r", "z")
        if "r_m" in final_particles.columns
        else ("x", "y", "z")[: int(spatial_dim)]
    )
    for axis in axes:
        column = f"{axis}_m"
        if column not in final_particles.columns:
            raise ValueError(
                f"final_particles.csv is missing position column {column!r}"
            )
        data.append(final_particles[column].to_numpy(dtype=np.float64))
    return np.stack(data, axis=1)


def _velocity_array(final_particles: pd.DataFrame, spatial_dim: int) -> np.ndarray:
    data = []
    axes = (
        ("r", "z")
        if "r_m" in final_particles.columns
        else ("x", "y", "z")[: int(spatial_dim)]
    )
    for axis in axes:
        column = f"v{axis}_mps"
        data.append(
            final_particles[column].to_numpy(dtype=np.float64)
            if column in final_particles.columns
            else np.full(len(final_particles), np.nan, dtype=np.float64)
        )
    return np.stack(data, axis=1)


def _normal_array(
    final_particles: pd.DataFrame,
    runtime: Any | None,
    positions: np.ndarray,
    spatial_dim: int,
) -> np.ndarray:
    data = []
    has_any = False
    axes = (
        ("r", "z")
        if "r_m" in final_particles.columns
        else ("x", "y", "z")[: int(spatial_dim)]
    )
    for axis in axes:
        name = f"contact_normal_{axis}"
        if name in final_particles.columns:
            data.append(final_particles[name].to_numpy(dtype=np.float64))
            has_any = True
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
            sampled.append(
                np.asarray(sample_geometry_normal(runtime, position), dtype=np.float64)[
                    : int(spatial_dim)
                ]
            )
        except Exception:
            sampled.append(np.full(int(spatial_dim), np.nan, dtype=np.float64))
    return np.asarray(sampled, dtype=np.float64)


def _column_or_nan(final_particles: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    for name in names:
        if name in final_particles.columns:
            return final_particles[name].to_numpy(dtype=np.float64)
    return np.full(len(final_particles), np.nan, dtype=np.float64)


def _column_or_int_zero(
    final_particles: pd.DataFrame, names: Sequence[str]
) -> np.ndarray:
    for name in names:
        if name in final_particles.columns:
            return final_particles[name].fillna(0).to_numpy(dtype=np.int64)
    return np.zeros(len(final_particles), dtype=np.int64)


def _inside_column(
    final_particles: pd.DataFrame, runtime: Any | None, positions: np.ndarray
) -> np.ndarray:
    if "inside_geometry" in final_particles.columns:
        return final_particles["inside_geometry"].to_numpy(dtype=np.float64)
    if runtime is None:
        return np.full(len(final_particles), np.nan, dtype=np.float64)
    values = []
    for position in positions:
        try:
            values.append(
                float(bool(inside_geometry(runtime, position, on_boundary_tol_m=0.0)))
            )
        except Exception:
            values.append(float("nan"))
    return np.asarray(values, dtype=np.float64)


def _sample_runtime_geometry(
    runtime: Any, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    sampled_sdf: list[float] = []
    sampled_part: list[int] = []
    for position in positions:
        try:
            sampled_sdf.append(float(sample_geometry_sdf(runtime, position)))
        except Exception:
            sampled_sdf.append(float("nan"))
        try:
            sampled_part.append(int(sample_geometry_part_id(runtime, position)))
        except Exception:
            sampled_part.append(0)
    return (
        np.asarray(sampled_sdf, dtype=np.float64),
        np.asarray(sampled_part, dtype=np.int64),
    )


def _prefer_runtime_geometry(
    runtime: Any,
    positions: np.ndarray,
    spatial_dim: int,
    *,
    sdf: np.ndarray,
    nearest_distance: np.ndarray,
    nearest_part: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled_sdf, sampled_part = _sample_runtime_geometry(runtime, positions)
    sdf = np.where(np.isfinite(sdf), sdf, sampled_sdf)
    nearest_part = np.where(nearest_part != 0, nearest_part, sampled_part)
    nearest_distance = np.where(
        np.isfinite(nearest_distance), nearest_distance, np.abs(sampled_sdf)
    )
    geometry_provider = getattr(runtime, "geometry_provider", None)
    geometry = (
        getattr(geometry_provider, "geometry", None)
        if geometry_provider is not None
        else None
    )
    if (
        int(spatial_dim) == 2
        and geometry is not None
        and getattr(geometry, "boundary_edges", None) is not None
    ):
        edge_part, edge_distance = nearest_boundary_edge_features_2d(runtime, positions)
        finite = np.isfinite(edge_distance)
        nearest_part = np.where(finite, edge_part.astype(np.int64), nearest_part)
        nearest_distance = np.where(finite, edge_distance, nearest_distance)
    return sdf, nearest_distance, nearest_part


def _normal_velocities(normals: np.ndarray, velocities: np.ndarray) -> np.ndarray:
    values = np.full(len(velocities), np.nan, dtype=np.float64)
    for idx, (normal, velocity) in enumerate(zip(normals, velocities, strict=True)):
        if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(velocity)):
            continue
        magnitude = float(np.linalg.norm(normal))
        if magnitude > 1.0e-30:
            values[idx] = float(np.dot(velocity, normal / magnitude))
    return values


def _geometry_features(
    final_particles: pd.DataFrame, runtime: Any | None, spatial_dim: int
) -> dict[str, np.ndarray]:
    positions = _position_array(final_particles, spatial_dim)
    velocities = _velocity_array(final_particles, spatial_dim)
    sdf = _column_or_nan(final_particles, ("sdf_m", "signed_sdf_m"))
    nearest_distance = _column_or_nan(
        final_particles, ("nearest_boundary_distance_m", "abs_sdf_m")
    )
    nearest_part = _column_or_int_zero(
        final_particles, ("nearest_boundary_part_id", "nearest_part_id", "part_id")
    )
    if runtime is not None:
        sdf, nearest_distance, nearest_part = _prefer_runtime_geometry(
            runtime,
            positions,
            spatial_dim,
            sdf=sdf,
            nearest_distance=nearest_distance,
            nearest_part=nearest_part,
        )
    normals = _normal_array(final_particles, runtime, positions, spatial_dim)
    return {
        "positions": positions,
        "velocities": velocities,
        "sdf_m": sdf,
        "nearest_boundary_distance_m": nearest_distance,
        "nearest_boundary_part_id": nearest_part.astype(np.int64),
        "inside_geometry": _inside_column(final_particles, runtime, positions),
        "normal_velocity_mps": _normal_velocities(normals, velocities),
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


def _positive_thresholds(
    source: Mapping[str, Any], names: Sequence[str]
) -> list[float]:
    values: list[float] = []
    for name in names:
        try:
            value = float(source.get(name, np.nan))
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value) and value > 0.0:
            values.append(value)
    return values


def _resolved_boundary_policy(report: Mapping[str, Any]) -> Mapping[str, Any]:
    execution = report.get("execution", {})
    if not isinstance(execution, Mapping):
        return {}
    numerics = execution.get("numerics", {})
    if not isinstance(numerics, Mapping):
        return {}
    boundary = numerics.get("boundary", {})
    return boundary if isinstance(boundary, Mapping) else {}


def _resolve_threshold(
    threshold_m: float | None, diagnostics: Mapping[str, Any], report: Mapping[str, Any]
) -> float:
    if threshold_m is not None:
        value = float(threshold_m)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("--threshold-m must be a finite non-negative value")
        return float(value)
    names = (
        "classification_tolerance_m",
        "contact_offset_m",
        "near_wall_threshold_m",
    )
    candidates: list[float] = []
    for source in (diagnostics, report):
        candidates.extend(_positive_thresholds(source, names))
    candidates.extend(
        _positive_thresholds(_resolved_boundary_policy(report), names[:2])
    )
    if not candidates:
        raise ValueError(
            "--threshold-m is required when run_summary.json does not contain "
            "the resolved boundary policy"
        )
    return float(max(candidates))


def _field_support_status(
    row: Mapping[str, Any], diagnostics: Mapping[str, Any], report: Mapping[str, Any]
) -> str:
    for name in ("field_support_status", "field_status", "valid_mask_status"):
        value = row.get(name, "")
        if value is not None and str(value).strip():
            return str(value)
    invalid_reason = str(row.get("invalid_stop_reason", "") or "").strip()
    if invalid_reason:
        return invalid_reason
    if (
        int(
            diagnostics.get(
                "valid_mask_hard_invalid_count",
                report.get("valid_mask_hard_invalid_count", 0),
            )
            or 0
        )
        > 0
    ):
        return "global_hard_invalid_seen"
    if (
        int(
            diagnostics.get(
                "valid_mask_mixed_stencil_count",
                report.get("valid_mask_mixed_stencil_count", 0),
            )
            or 0
        )
        > 0
    ):
        return "global_mixed_stencil_seen"
    return "unknown"


def _is_field_support_issue(status: str) -> bool:
    text = str(status).strip().lower()
    return any(
        token in text
        for token in ("field", "valid_mask", "hard_invalid", "mixed_stencil", "support")
    )


def _classify_row(
    *,
    row: Mapping[str, Any],
    wall_events_available: bool,
    diagnostics: Mapping[str, Any],
    field_support_status: str,
) -> tuple[str, str]:
    if not wall_events_available:
        return (
            "no_wall_events_available",
            "wall_events.csv was not present, so per-particle hit history is "
            "unavailable",
        )
    if _is_field_support_issue(field_support_status):
        return (
            "field_support_issue",
            f"field support status is {field_support_status!r}",
        )
    if (
        str(row.get("final_state", "")) == "numerical_boundary_stopped"
        or int(diagnostics.get("unresolved_crossing_count", 0) or 0) > 0
    ):
        return (
            "unresolved_crossing_numerical_boundary_issue",
            "numerical boundary or unresolved crossing diagnostics are present",
        )
    return (
        "no_segment_crossing_recorded",
        "particle is active near a boundary but has no recorded wall hit",
    )


def _top_counts(values: Sequence[Any], key_name: str) -> list[dict[str, Any]]:
    counts = pd.Series(list(values), dtype=object).value_counts(dropna=False)
    return [
        {key_name: str(key), "count": int(value)}
        for key, value in counts.head(12).items()
    ]


def _load_runtime(config_path: Path | None) -> Any | None:
    if config_path is None:
        return None
    return load_case(Path(config_path).resolve()).solver_context


def _prepare_analysis_paths(
    output_dir: Path, analysis_output_dir: Path
) -> tuple[Path, Path]:
    output = Path(output_dir).resolve()
    analysis = Path(analysis_output_dir).resolve()
    if analysis == output or output in analysis.parents:
        raise ValueError(
            "analysis_output_dir must be separate from the immutable solver "
            "output directory"
        )
    analysis.mkdir(parents=True, exist_ok=True)
    return output, analysis


def _load_analysis_inputs(
    output_dir: Path, config_path: Path | None
) -> tuple[pd.DataFrame, Mapping[str, Any], Mapping[str, Any], Any | None]:
    final_path = output_dir / "final_particles.csv"
    if not final_path.exists():
        raise FileNotFoundError(final_path)
    final_particles = pd.read_csv(final_path)
    if "particle_id" not in final_particles.columns:
        raise ValueError("final_particles.csv is missing required column 'particle_id'")
    diagnostics = _load_json_optional(output_dir / "debug_diagnostics.json").get(
        "collision", {}
    )
    report = _load_json_optional(output_dir / "run_summary.json")
    return final_particles, diagnostics, report, _load_runtime(config_path)


def _near_wall_masks(
    states: np.ndarray,
    features: Mapping[str, np.ndarray],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active = np.isin(states, ("active_free_flight", "contact_sliding"))
    distance = np.asarray(features["nearest_boundary_distance_m"], dtype=np.float64)
    sdf = np.asarray(features["sdf_m"], dtype=np.float64)
    near = (np.isfinite(distance) & (distance <= threshold)) | (
        np.isfinite(sdf) & (np.abs(sdf) <= threshold)
    )
    return active, near, distance, sdf


def _vector_component(values: np.ndarray, index: int, component: int) -> float:
    if component >= values.shape[1]:
        return float("nan")
    return float(values[index, component])


def _analysis_row(
    *,
    row_index: int,
    raw_row: Mapping[str, Any],
    states: np.ndarray,
    features: Mapping[str, np.ndarray],
    distance: np.ndarray,
    sdf: np.ndarray,
    wall_events_available: bool,
    wall_hit_count: int,
    diagnostics: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    particle_id = _int_or_default(raw_row.get("particle_id", 0), 0)
    field_status = _field_support_status(raw_row, diagnostics, report)
    classification, reason = _classify_row(
        row=raw_row,
        wall_events_available=wall_events_available,
        diagnostics=diagnostics,
        field_support_status=field_status,
    )
    positions = features["positions"]
    velocities = features["velocities"]
    inside_value = float(features["inside_geometry"][row_index])
    return {
        "particle_id": particle_id,
        "source_part_id": _int_or_default(raw_row.get("source_part_id", 0), 0),
        "final_state_class": str(states[row_index]),
        "x": _vector_component(positions, row_index, 0),
        "y": _vector_component(positions, row_index, 1),
        "z": _vector_component(positions, row_index, 2),
        "v_x": _vector_component(velocities, row_index, 0),
        "v_y": _vector_component(velocities, row_index, 1),
        "v_z": _vector_component(velocities, row_index, 2),
        "nearest_boundary_part_id": int(
            features["nearest_boundary_part_id"][row_index]
        ),
        "nearest_boundary_distance_m": float(distance[row_index]),
        "sdf_m": float(sdf[row_index]),
        "inside_geometry": int(inside_value) if np.isfinite(inside_value) else "",
        "normal_velocity_mps": float(features["normal_velocity_mps"][row_index]),
        "wall_hit_count": wall_hit_count,
        "wall_events_available": int(wall_events_available),
        "field_support_status": field_status,
        "classification": classification,
        "classification_reason": reason,
    }


def _suspicious_rows(
    final_particles: pd.DataFrame,
    *,
    states: np.ndarray,
    features: Mapping[str, np.ndarray],
    active_mask: np.ndarray,
    near_mask: np.ndarray,
    distance: np.ndarray,
    sdf: np.ndarray,
    wall_events_available: bool,
    wall_counts: Mapping[int, int],
    diagnostics: Mapping[str, Any],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, (_, raw_row) in enumerate(final_particles.iterrows()):
        row_values = raw_row.to_dict()
        particle_id = _int_or_default(row_values.get("particle_id", 0), 0)
        wall_hit_count = int(wall_counts.get(particle_id, 0))
        if (
            not bool(active_mask[row_index])
            or not bool(near_mask[row_index])
            or (wall_events_available and wall_hit_count > 0)
        ):
            continue
        rows.append(
            _analysis_row(
                row_index=row_index,
                raw_row=row_values,
                states=states,
                features=features,
                distance=distance,
                sdf=sdf,
                wall_events_available=wall_events_available,
                wall_hit_count=wall_hit_count,
                diagnostics=diagnostics,
                report=report,
            )
        )
    return rows


def _classification_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    if not rows:
        return {}
    counts = pd.Series(
        [row["classification"] for row in rows], dtype=object
    ).value_counts()
    return {str(key): int(value) for key, value in counts.items()}


def _analysis_summary(
    *,
    output_dir: Path,
    analysis_dir: Path,
    config_path: Path | None,
    threshold: float,
    runtime: Any | None,
    final_particles: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    wall_events_available: bool,
    active_mask: np.ndarray,
    near_mask: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    particles_path: Path,
) -> dict[str, Any]:
    geometry_available = (
        runtime is not None
        or "nearest_boundary_distance_m" in final_particles.columns
        or "sdf_m" in final_particles.columns
    )
    return {
        "output_dir": str(output_dir),
        "config": None if config_path is None else str(Path(config_path)),
        "threshold_m": threshold,
        "geometry_available": int(geometry_available),
        "wall_events_available": int(wall_events_available),
        "collision_diagnostics_available": int(bool(diagnostics)),
        "final_particle_count": len(final_particles),
        "active_particle_count": int(np.count_nonzero(active_mask)),
        "near_wall_active_count": int(np.count_nonzero(active_mask & near_mask)),
        "suspicious_particle_count": len(rows),
        "classification_counts": _classification_counts(rows),
        "nearest_boundary_part_counts": _top_counts(
            [row["nearest_boundary_part_id"] for row in rows], "part_id"
        ),
        "artifacts": {
            "near_wall_nohit_particles_csv": str(particles_path),
            "near_wall_nohit_summary_json": str(
                analysis_dir / "near_wall_nohit_summary.json"
            ),
        },
        "interpretation": (
            "Rows are active particles near a boundary with no recorded wall hit. "
            "Use classification together with source_part_id, nearest part, "
            "SDF/inside status, normal velocity, safety counters, and field support "
            "status to decide whether the next fix belongs in release semantics, "
            "segment crossing, field support, source provenance, or "
            "integrator/force behavior."
        ),
    }


def analyze_near_wall_nohit(
    *,
    output_dir: Path,
    config_path: Path | None = None,
    threshold_m: float | None = None,
    analysis_output_dir: Path,
) -> dict[str, Any]:
    output_dir, analysis_dir = _prepare_analysis_paths(output_dir, analysis_output_dir)
    final_particles, diagnostics, report, runtime = _load_analysis_inputs(
        output_dir, config_path
    )
    spatial_dim = _spatial_dim(final_particles, runtime)
    threshold = _resolve_threshold(threshold_m, diagnostics, report)
    features = _geometry_features(final_particles, runtime, spatial_dim)
    states = _final_state_labels(final_particles)
    wall_events_available, wall_counts = _wall_hit_counts(output_dir)
    active_mask, near_mask, distance, sdf = _near_wall_masks(
        states, features, threshold
    )
    rows = _suspicious_rows(
        final_particles,
        states=states,
        features=features,
        active_mask=active_mask,
        near_mask=near_mask,
        distance=distance,
        sdf=sdf,
        wall_events_available=wall_events_available,
        wall_counts=wall_counts,
        diagnostics=diagnostics,
        report=report,
    )

    particles_path = analysis_dir / "near_wall_nohit_particles.csv"
    pd.DataFrame(rows, columns=NEAR_WALL_NOHIT_COLUMNS).to_csv(
        particles_path, index=False
    )
    summary = _analysis_summary(
        output_dir=output_dir,
        analysis_dir=analysis_dir,
        config_path=config_path,
        threshold=threshold,
        runtime=runtime,
        final_particles=final_particles,
        diagnostics=diagnostics,
        wall_events_available=wall_events_available,
        active_mask=active_mask,
        near_mask=near_mask,
        rows=rows,
        particles_path=particles_path,
    )
    summary_path = analysis_dir / "near_wall_nohit_summary.json"
    summary_path.write_text(
        json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="particle-tracer compare near-wall",
        description=(
            "Analyze active near-wall final particles that have no recorded wall hit."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Existing solver output directory",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional run_config.yaml used to sample geometry",
    )
    parser.add_argument(
        "--threshold-m",
        type=float,
        default=None,
        help=(
            "Near-wall distance threshold. Defaults to the resolved boundary "
            "policy in run_summary.json."
        ),
    )
    parser.add_argument(
        "--analysis-output-dir",
        type=Path,
        required=True,
        help="Separate directory for the two derived analysis artifacts.",
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
