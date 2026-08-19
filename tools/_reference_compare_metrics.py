from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from particle_tracer_unified.core.boundary_service import (
    nearest_boundary_edge_features_2d,
    sample_geometry_part_id,
    sample_geometry_sdf,
)
from tools.state_contract import particle_class_frame


def _merged_particle_classes(
    candidate_final: pd.DataFrame,
    reference_final: pd.DataFrame,
) -> pd.DataFrame:
    return particle_class_frame(candidate_final).merge(
        particle_class_frame(reference_final),
        on="particle_id",
        how="inner",
        suffixes=("_candidate", "_reference"),
    )


def class_match_ratio(
    candidate_final: pd.DataFrame, reference_final: pd.DataFrame
) -> tuple[float, int]:
    merged = _merged_particle_classes(candidate_final, reference_final)
    if merged.empty:
        return 0.0, 0
    matches = merged["particle_class_candidate"].astype(str) == merged[
        "particle_class_reference"
    ].astype(str)
    return float(matches.mean()), len(merged)


def class_transition_summary(
    candidate_final: pd.DataFrame, reference_final: pd.DataFrame, *, top_n: int = 12
) -> dict[str, Any]:
    merged = _merged_particle_classes(candidate_final, reference_final)
    if merged.empty:
        return {"compared_particles": 0, "mismatch_count": 0, "top_transitions": []}
    matches = merged["particle_class_candidate"].astype(str) == merged[
        "particle_class_reference"
    ].astype(str)
    transitions = (
        merged.groupby(["particle_class_reference", "particle_class_candidate"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    top_rows = [
        {
            "reference_class": str(row["particle_class_reference"]),
            "candidate_class": str(row["particle_class_candidate"]),
            "count": int(row["count"]),
        }
        for _, row in transitions.head(int(top_n)).iterrows()
    ]
    mismatch_rows = transitions[
        transitions["particle_class_reference"]
        != transitions["particle_class_candidate"]
    ]
    top_mismatches = [
        {
            "reference_class": str(row["particle_class_reference"]),
            "candidate_class": str(row["particle_class_candidate"]),
            "count": int(row["count"]),
        }
        for _, row in mismatch_rows.head(int(top_n)).iterrows()
    ]
    return {
        "compared_particles": len(merged),
        "mismatch_count": int((~matches).sum()),
        "top_transitions": top_rows,
        "top_mismatches": top_mismatches,
    }


def _finite_summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p50": float(np.percentile(finite, 50.0)),
        "p90": float(np.percentile(finite, 90.0)),
        "p99": float(np.percentile(finite, 99.0)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _final_spatial_dim(final_df: pd.DataFrame) -> int:
    if {"r_m", "z_m"}.issubset(final_df.columns):
        return 2
    return int(sum(1 for name in ("x_m", "y_m", "z_m") if name in final_df.columns))


def final_position_array(final_df: pd.DataFrame, spatial_dim: int) -> np.ndarray:
    names = (
        ["r_m", "z_m"]
        if "r_m" in final_df.columns
        else ["x_m", "y_m", "z_m"][: int(spatial_dim)]
    )
    missing = [name for name in names if name not in final_df.columns]
    if missing:
        raise ValueError(f"final_particles.csv is missing position columns: {missing}")
    return final_df[names].to_numpy(dtype=np.float64)


def _final_velocity_array(final_df: pd.DataFrame, spatial_dim: int) -> np.ndarray:
    values: list[np.ndarray] = []
    axes = ["r", "z"] if "r_m" in final_df.columns else ["x", "y", "z"][:spatial_dim]
    for name in axes:
        column = f"v{name}_mps"
        values.append(
            final_df[column].to_numpy(dtype=np.float64)
            if column in final_df.columns
            else np.zeros(len(final_df), dtype=np.float64)
        )
    if not values:
        return np.zeros((len(final_df), 0), dtype=np.float64)
    return np.stack(values, axis=1)


def _feature_near_boundary_threshold_m(runtime: Any) -> float:
    geometry_provider = getattr(runtime, "geometry_provider", None)
    if geometry_provider is None:
        return 0.0
    spacings: list[float] = []
    for axis in getattr(geometry_provider.geometry, "axes", ()):
        values = np.asarray(axis, dtype=np.float64)
        diffs = np.diff(values)
        positive = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if positive.size:
            spacings.append(float(np.min(positive)))
    return float(min(spacings)) if spacings else 0.0


def _geometry_feature_frame(runtime: Any, final_df: pd.DataFrame) -> pd.DataFrame:
    spatial_dim = int(getattr(runtime, "spatial_dim", _final_spatial_dim(final_df)))
    spatial_dim = min(spatial_dim, _final_spatial_dim(final_df))
    positions = final_position_array(final_df, spatial_dim)
    velocities = _final_velocity_array(final_df, spatial_dim)
    sdf_values = np.asarray(
        [sample_geometry_sdf(runtime, position) for position in positions],
        dtype=np.float64,
    )
    nearest_part_ids = np.asarray(
        [sample_geometry_part_id(runtime, position) for position in positions],
        dtype=np.int32,
    )
    nearest_distances = np.abs(sdf_values)
    geometry_provider = getattr(runtime, "geometry_provider", None)
    if (
        spatial_dim == 2
        and geometry_provider is not None
        and getattr(geometry_provider.geometry, "boundary_edges", None) is not None
    ):
        edge_part_ids, edge_distances = nearest_boundary_edge_features_2d(
            runtime, positions
        )
        finite_edge = np.isfinite(edge_distances)
        if np.any(finite_edge):
            nearest_part_ids = np.where(finite_edge, edge_part_ids, nearest_part_ids)
            nearest_distances = np.where(finite_edge, edge_distances, nearest_distances)

    frame = pd.DataFrame(
        {
            "particle_id": final_df["particle_id"].astype(np.int64),
            "sdf_m": sdf_values,
            "abs_sdf_m": np.abs(sdf_values),
            "nearest_boundary_distance_m": nearest_distances,
            "nearest_part_id": nearest_part_ids.astype(np.int32),
            "speed_mps": (
                np.linalg.norm(velocities, axis=1)
                if velocities.size
                else np.zeros(len(final_df), dtype=np.float64)
            ),
        }
    )
    axis_names = (
        ["r_m", "z_m"]
        if "r_m" in final_df.columns
        else ["x_m", "y_m", "z_m"][:spatial_dim]
    )
    for axis_index, axis_name in enumerate(axis_names):
        frame[axis_name] = positions[:, axis_index]
    return frame.merge(particle_class_frame(final_df), on="particle_id", how="left")


def _top_part_transitions(
    merged: pd.DataFrame, *, top_n: int = 12
) -> list[dict[str, Any]]:
    transitions = (
        merged.groupby(["nearest_part_id_reference", "nearest_part_id_candidate"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return [
        {
            "reference_part_id": int(row["nearest_part_id_reference"]),
            "candidate_part_id": int(row["nearest_part_id_candidate"]),
            "count": int(row["count"]),
        }
        for _, row in transitions.head(int(top_n)).iterrows()
    ]


def _position_errors(merged: pd.DataFrame) -> np.ndarray:
    coordinate_names = (
        ["r_m", "z_m"] if "r_m_reference" in merged.columns else ["x_m", "y_m", "z_m"]
    )
    shared_names = [
        name
        for name in coordinate_names
        if f"{name}_reference" in merged.columns
        and f"{name}_candidate" in merged.columns
    ]
    if not shared_names:
        return np.full(len(merged), np.nan, dtype=np.float64)
    reference = merged[[f"{name}_reference" for name in shared_names]].to_numpy(
        dtype=np.float64
    )
    candidate = merged[[f"{name}_candidate" for name in shared_names]].to_numpy(
        dtype=np.float64
    )
    return np.linalg.norm(candidate - reference, axis=1)


def _near_boundary_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0.0:
        return np.zeros(len(values), dtype=bool)
    return np.isfinite(values) & (np.abs(values) <= threshold)


def _outside_geometry_count(values: np.ndarray) -> int:
    return int(np.count_nonzero(np.isfinite(values) & (values > 0.0)))


def _mismatched_feature_summary(
    mismatch_mask: np.ndarray,
    *,
    position_error: np.ndarray,
    sdf_candidate: np.ndarray,
    sdf_reference: np.ndarray,
    distance_candidate: np.ndarray,
    distance_reference: np.ndarray,
) -> dict[str, Any]:
    return {
        "count": int(np.count_nonzero(mismatch_mask)),
        "position_error_m": _finite_summary(position_error[mismatch_mask]),
        "candidate_sdf_m": _finite_summary(sdf_candidate[mismatch_mask]),
        "reference_sdf_m": _finite_summary(sdf_reference[mismatch_mask]),
        "candidate_nearest_boundary_distance_m": _finite_summary(
            distance_candidate[mismatch_mask]
        ),
        "reference_nearest_boundary_distance_m": _finite_summary(
            distance_reference[mismatch_mask]
        ),
    }


def geometry_feature_delta_summary(
    candidate_final: pd.DataFrame,
    reference_final: pd.DataFrame,
    runtime,
    *,
    top_n: int = 12,
) -> dict[str, Any]:
    candidate_features = _geometry_feature_frame(runtime, candidate_final)
    reference_features = _geometry_feature_frame(runtime, reference_final)
    merged = reference_features.merge(
        candidate_features,
        on="particle_id",
        how="inner",
        suffixes=("_reference", "_candidate"),
    )
    if merged.empty:
        return {"compared_particles": 0}

    position_error = _position_errors(merged)
    sdf_reference = merged["sdf_m_reference"].to_numpy(dtype=np.float64)
    sdf_candidate = merged["sdf_m_candidate"].to_numpy(dtype=np.float64)
    distance_reference = merged["nearest_boundary_distance_m_reference"].to_numpy(
        dtype=np.float64
    )
    distance_candidate = merged["nearest_boundary_distance_m_candidate"].to_numpy(
        dtype=np.float64
    )
    speed_reference = merged["speed_mps_reference"].to_numpy(dtype=np.float64)
    speed_candidate = merged["speed_mps_candidate"].to_numpy(dtype=np.float64)
    threshold = _feature_near_boundary_threshold_m(runtime)
    reference_near = _near_boundary_mask(sdf_reference, threshold)
    candidate_near = _near_boundary_mask(sdf_candidate, threshold)
    class_matches = (
        merged["particle_class_reference"].astype(str).to_numpy()
        == merged["particle_class_candidate"].astype(str).to_numpy()
    )
    mismatch_mask = np.logical_not(class_matches)
    outside_reference = _outside_geometry_count(sdf_reference)
    outside_candidate = _outside_geometry_count(sdf_candidate)
    near_reference = int(np.count_nonzero(reference_near))
    near_candidate = int(np.count_nonzero(candidate_near))

    return {
        "compared_particles": len(merged),
        "near_boundary_threshold_m": float(threshold),
        "position_error_m": _finite_summary(position_error),
        "sdf_error_m": _finite_summary(np.abs(sdf_candidate - sdf_reference)),
        "abs_sdf_error_m": _finite_summary(
            np.abs(np.abs(sdf_candidate) - np.abs(sdf_reference))
        ),
        "nearest_boundary_distance_error_m": _finite_summary(
            np.abs(distance_candidate - distance_reference)
        ),
        "speed_error_mps": _finite_summary(np.abs(speed_candidate - speed_reference)),
        "outside_geometry_count_reference": outside_reference,
        "outside_geometry_count_candidate": outside_candidate,
        "outside_geometry_count_delta": outside_candidate - outside_reference,
        "near_boundary_count_reference": near_reference,
        "near_boundary_count_candidate": near_candidate,
        "near_boundary_count_delta": near_candidate - near_reference,
        "nearest_part_transition_summary": _top_part_transitions(merged, top_n=top_n),
        "mismatched_state_feature_summary": _mismatched_feature_summary(
            mismatch_mask,
            position_error=position_error,
            sdf_candidate=sdf_candidate,
            sdf_reference=sdf_reference,
            distance_candidate=distance_candidate,
            distance_reference=distance_reference,
        ),
    }


def pair_delta(
    base_run: Mapping[str, Any], candidate_run: Mapping[str, Any]
) -> dict[str, Any]:
    base_runtime = float(base_run.get("runtime_s", 0.0))
    candidate_runtime = float(candidate_run.get("runtime_s", 0.0))
    runtime_increase_ratio = (
        0.0
        if base_runtime <= 0.0
        else (candidate_runtime - base_runtime) / base_runtime
    )
    return {
        "base_run": str(base_run.get("run", "")),
        "candidate_run": str(candidate_run.get("run", "")),
        "runtime_increase_ratio": float(runtime_increase_ratio),
        "class_match_ratio_delta": float(
            candidate_run.get("class_match_ratio_vs_reference", 0.0)
            - base_run.get("class_match_ratio_vs_reference", 0.0)
        ),
        "unresolved_crossing_count_delta": int(
            candidate_run.get("unresolved_crossing_count", 0)
            - base_run.get("unresolved_crossing_count", 0)
        ),
        "max_hits_reached_count_delta": int(
            candidate_run.get("max_hits_reached_count", 0)
            - base_run.get("max_hits_reached_count", 0)
        ),
        "nearest_projection_fallback_count_delta": int(
            candidate_run.get("nearest_projection_fallback_count", 0)
            - base_run.get("nearest_projection_fallback_count", 0)
        ),
        "boundary_event_failure_count_delta": int(
            candidate_run.get("boundary_event_failure_count", 0)
            - base_run.get("boundary_event_failure_count", 0)
        ),
        "stuck_count_delta": int(
            candidate_run.get("stuck_count", 0) - base_run.get("stuck_count", 0)
        ),
        "invalid_mask_stopped_count_delta": int(
            candidate_run.get("invalid_mask_stopped_count", 0)
            - base_run.get("invalid_mask_stopped_count", 0)
        ),
        "valid_mask_mixed_stencil_count_delta": int(
            candidate_run.get("valid_mask_mixed_stencil_count", 0)
            - base_run.get("valid_mask_mixed_stencil_count", 0)
        ),
        "valid_mask_hard_invalid_count_delta": int(
            candidate_run.get("valid_mask_hard_invalid_count", 0)
            - base_run.get("valid_mask_hard_invalid_count", 0)
        ),
    }


def _diagnostic_failure_names(runs: Iterable[Mapping[str, Any]]) -> list[str]:
    return [
        str(run.get("run", ""))
        for run in runs
        if bool(run.get("diagnostic_hard_invalid_failed", False))
    ]


def _boundary_failure_names(runs: Iterable[Mapping[str, Any]]) -> list[str]:
    return [
        str(run.get("run", ""))
        for run in runs
        if int(run.get("boundary_event_failure_count", 0)) > 0
    ]


def comparison_summary(
    args: argparse.Namespace,
    *,
    timestamp: str,
    comparison_dir: Path,
    reference: dict[str, Any],
    runs: list[dict[str, Any]],
) -> tuple[dict[str, Any], int]:
    summary: dict[str, Any] = {
        "artifact_type": "particle_tracer.reference_comparison",
        "schema_version": 2,
        "timestamp": timestamp,
        "comparison_dir": str(comparison_dir),
        "reference_scope": str(args.reference_scope),
        "overrides": {
            "t_end": (
                None if args.override_t_end is None else float(args.override_t_end)
            ),
            "artifact_mode": args.artifact_mode,
        },
        "reference": reference,
        "runs": runs,
    }
    if len(runs) == 2:
        summary["pair_delta"] = pair_delta(runs[0], runs[1])
    all_runs = [reference, *runs]
    diagnostic_failures = _diagnostic_failure_names(all_runs)
    boundary_failures = _boundary_failure_names(all_runs)
    if diagnostic_failures:
        summary["diagnostic_hard_invalid_failures"] = diagnostic_failures
    if boundary_failures:
        summary["boundary_event_failures"] = boundary_failures
    return summary, int(bool(diagnostic_failures or boundary_failures))
