from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from particle_tracer_unified.core.boundary_service import inside_geometry
from particle_tracer_unified.core.field_backend import sample_field_valid_status
from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
)
from particle_tracer_unified.io.runtime_builder import build_runtime_from_config

import yaml


STATUS_NAMES = {
    int(VALID_MASK_STATUS_CLEAN): "clean",
    int(VALID_MASK_STATUS_MIXED_STENCIL): "mixed_stencil",
    int(VALID_MASK_STATUS_HARD_INVALID): "hard_invalid",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML object expected: {path}")
    return payload


def _field_cell_diagonal_m(runtime: Any) -> float:
    field_provider = getattr(runtime, "field_provider", None)
    if field_provider is None:
        raise ValueError("A field provider is required to classify boundary release support")
    field = field_provider.field
    axes = tuple(getattr(field, "axes", ()))
    steps: list[float] = []
    for axis in axes:
        values = np.asarray(axis, dtype=np.float64)
        diff = np.diff(values)
        positive = diff[np.isfinite(diff) & (diff > 0.0)]
        if positive.size:
            steps.append(float(np.max(positive)))
    if not steps:
        raise ValueError("Cannot infer field cell size from field axes")
    return float(np.sqrt(np.sum(np.square(steps))))


def _nearest_boundary_frame(runtime: Any, point: np.ndarray, source_part_id: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    geometry_provider = getattr(runtime, "geometry_provider", None)
    if geometry_provider is None:
        raise ValueError("A geometry provider is required to offset boundary release particles")
    geom = geometry_provider.geometry
    edges = np.asarray(getattr(geom, "boundary_edges", None), dtype=np.float64)
    part_ids = np.asarray(getattr(geom, "boundary_edge_part_ids", None), dtype=np.int64)
    if edges.ndim != 3 or edges.shape[1:] != (2, 2):
        raise ValueError("Only 2D boundary edge geometry is supported by this tool")
    if part_ids.size != edges.shape[0]:
        part_ids = np.zeros(edges.shape[0], dtype=np.int64)

    candidates = np.flatnonzero(part_ids == int(source_part_id)) if int(source_part_id) > 0 else np.asarray([], dtype=np.int64)
    if candidates.size == 0:
        candidates = np.arange(edges.shape[0], dtype=np.int64)
    a = edges[candidates, 0, :]
    b = edges[candidates, 1, :]
    ab = b - a
    denom = np.maximum(np.sum(ab * ab, axis=1), 1.0e-300)
    ap = np.asarray(point, dtype=np.float64)[None, :] - a
    alpha = np.clip(np.sum(ap * ab, axis=1) / denom, 0.0, 1.0)
    closest = a + alpha[:, None] * ab
    delta = np.asarray(point, dtype=np.float64)[None, :] - closest
    d2 = np.sum(delta * delta, axis=1)
    local = int(np.argmin(d2))
    edge_index = int(candidates[local])
    tangent = ab[local]
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1.0e-30:
        raise ValueError("Degenerate boundary edge encountered")
    tangent = tangent / tangent_norm
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    return tangent, normal, int(part_ids[edge_index]), float(np.sqrt(max(float(d2[local]), 0.0)))


def _offset_candidates(
    point: np.ndarray,
    tangent: np.ndarray,
    normal: np.ndarray,
    *,
    base_offset_m: float,
    max_normal_cells: float,
    max_tangent_cells: float,
) -> Iterable[tuple[np.ndarray, float, float, int]]:
    normal_cells = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    tangent_cells = [0.0, -0.25, 0.25, -0.5, 0.5, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0]
    normal_values = [v for v in normal_cells if v <= float(max_normal_cells) + 1.0e-12]
    tangent_values = [v for v in tangent_cells if abs(v) <= float(max_tangent_cells) + 1.0e-12]
    if not normal_values:
        normal_values = [float(max_normal_cells)]
    if not tangent_values:
        tangent_values = [0.0]
    rows: list[tuple[float, np.ndarray, float, float, int]] = []
    p = np.asarray(point, dtype=np.float64)
    for n_cells in normal_values:
        n_offset = float(n_cells) * float(base_offset_m)
        for t_cells in tangent_values:
            t_offset = float(t_cells) * float(base_offset_m)
            for sign in (-1, 1):
                candidate = p + float(sign) * n_offset * normal + t_offset * tangent
                displacement = float(np.linalg.norm(candidate - p))
                rows.append((displacement, candidate, n_offset, t_offset, int(sign)))
    rows.sort(key=lambda row: row[0])
    for _displacement, candidate, n_offset, t_offset, sign in rows:
        yield candidate, n_offset, t_offset, sign


def offset_boundary_release_particles(
    config_yaml: str | Path,
    out_csv: str | Path,
    *,
    particles_csv: str | Path | None = None,
    report_json: str | Path | None = None,
    offset_cells: float = 1.0,
    offset_m: float | None = None,
    max_normal_offset_cells: float = 4.0,
    max_tangent_offset_cells: float = 2.0,
    preserve_clean: bool = True,
) -> dict[str, Any]:
    config_path = Path(config_yaml).resolve()
    config = _read_yaml(config_path)
    if particles_csv is not None:
        config = dict(config)
        config["paths"] = dict(config.get("paths", {}))
        input_particles = Path(particles_csv)
        config["paths"]["particles_csv"] = str(input_particles if input_particles.is_absolute() else input_particles)
    runtime = build_runtime_from_config(config, config_path.parent)
    if runtime.particles is None:
        raise ValueError("Runtime does not contain particles")
    if runtime.field_provider is None:
        raise ValueError("Runtime does not contain a field provider")

    input_path = Path(particles_csv) if particles_csv is not None else config_path.parent / str(config.get("paths", {}).get("particles_csv"))
    input_path = input_path if input_path.is_absolute() else (config_path.parent / input_path).resolve()
    frame = pd.read_csv(input_path)
    positions = np.asarray(runtime.particles.position[:, :2], dtype=np.float64)
    release_times = np.asarray(runtime.particles.release_time, dtype=np.float64)
    source_part_ids = np.asarray(runtime.particles.source_part_id, dtype=np.int64)
    base_offset = float(offset_m) if offset_m is not None else float(offset_cells) * _field_cell_diagonal_m(runtime)

    diagnostics: list[dict[str, Any]] = []
    adjusted = positions.copy()
    failed = 0
    moved = 0
    already_clean = 0
    for i, point in enumerate(positions):
        t_eval = float(release_times[i])
        source_part_id = int(source_part_ids[i])
        initial_status = int(sample_field_valid_status(runtime.field_provider, point, t_eval))
        if bool(preserve_clean) and initial_status == int(VALID_MASK_STATUS_CLEAN):
            already_clean += 1
            diagnostics.append(
                {
                    "particle_id": int(runtime.particles.particle_id[i]),
                    "source_part_id": source_part_id,
                    "initial_status": STATUS_NAMES.get(initial_status, "unknown"),
                    "final_status": STATUS_NAMES.get(initial_status, "unknown"),
                    "moved": 0,
                    "offset_distance_m": 0.0,
                    "normal_offset_m": 0.0,
                    "tangent_offset_m": 0.0,
                    "normal_sign": 0,
                    "nearest_boundary_part_id": source_part_id,
                    "nearest_boundary_distance_m": 0.0,
                }
            )
            continue

        tangent, normal, nearest_part, nearest_distance = _nearest_boundary_frame(runtime, point, source_part_id)
        chosen: tuple[np.ndarray, float, float, int, int] | None = None
        for candidate, normal_offset, tangent_offset, normal_sign in _offset_candidates(
            point,
            tangent,
            normal,
            base_offset_m=base_offset,
            max_normal_cells=float(max_normal_offset_cells),
            max_tangent_cells=float(max_tangent_offset_cells),
        ):
            if not inside_geometry(runtime, candidate, on_boundary_tol_m=0.0):
                continue
            status = int(sample_field_valid_status(runtime.field_provider, candidate, t_eval))
            if status == int(VALID_MASK_STATUS_CLEAN):
                chosen = (candidate, normal_offset, tangent_offset, normal_sign, status)
                break

        if chosen is None:
            failed += 1
            final_status = initial_status
            normal_offset = 0.0
            tangent_offset = 0.0
            normal_sign = 0
            candidate = point
        else:
            candidate, normal_offset, tangent_offset, normal_sign, final_status = chosen
            adjusted[i, :] = candidate
            moved += 1

        diagnostics.append(
            {
                "particle_id": int(runtime.particles.particle_id[i]),
                "source_part_id": source_part_id,
                "initial_status": STATUS_NAMES.get(initial_status, "unknown"),
                "final_status": STATUS_NAMES.get(int(final_status), "unknown"),
                "moved": int(chosen is not None),
                "offset_distance_m": float(np.linalg.norm(np.asarray(candidate, dtype=np.float64) - point)),
                "normal_offset_m": float(normal_offset),
                "tangent_offset_m": float(tangent_offset),
                "normal_sign": int(normal_sign),
                "nearest_boundary_part_id": int(nearest_part),
                "nearest_boundary_distance_m": float(nearest_distance),
            }
        )

    output = frame.copy()
    output["x"] = adjusted[:, 0]
    y_col = "y" if "y" in output.columns else "z"
    output[y_col] = adjusted[:, 1]
    output["release_x_original"] = positions[:, 0]
    output["release_y_original"] = positions[:, 1]
    output["release_offset_distance_m"] = np.linalg.norm(adjusted - positions, axis=1)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)
    diag_frame = pd.DataFrame(diagnostics)
    diag_path = out_path.with_name(out_path.stem + "_offset_diagnostics.csv")
    diag_frame.to_csv(diag_path, index=False)
    final_counts = diag_frame["final_status"].value_counts().sort_index().to_dict() if not diag_frame.empty else {}
    initial_counts = diag_frame["initial_status"].value_counts().sort_index().to_dict() if not diag_frame.empty else {}
    displacement = output["release_offset_distance_m"].to_numpy(dtype=np.float64)
    moved_displacement = displacement[displacement > 0.0]
    report = {
        "source_kind": "boundary_release_inward_clean_particles",
        "config_yaml": str(config_path),
        "input_particles_csv": str(input_path),
        "out_csv": str(out_path),
        "diagnostics_csv": str(diag_path),
        "particle_count": int(len(output)),
        "base_offset_m": float(base_offset),
        "offset_cells": float(offset_cells),
        "max_normal_offset_cells": float(max_normal_offset_cells),
        "max_tangent_offset_cells": float(max_tangent_offset_cells),
        "preserve_clean": bool(preserve_clean),
        "already_clean_count": int(already_clean),
        "moved_count": int(moved),
        "failed_count": int(failed),
        "initial_status_counts": {str(k): int(v) for k, v in initial_counts.items()},
        "final_status_counts": {str(k): int(v) for k, v in final_counts.items()},
        "offset_distance_m": {
            "max": float(np.max(displacement)) if displacement.size else 0.0,
            "mean_moved": float(np.mean(moved_displacement)) if moved_displacement.size else 0.0,
            "median_moved": float(np.median(moved_displacement)) if moved_displacement.size else 0.0,
        },
    }
    if report_json is not None:
        report_path = Path(report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a reported inward-clean particle table for boundary-release comparisons.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--particles-csv", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--offset-cells", type=float, default=1.0)
    parser.add_argument("--offset-m", type=float, default=None)
    parser.add_argument("--max-normal-offset-cells", type=float, default=4.0)
    parser.add_argument("--max-tangent-offset-cells", type=float, default=2.0)
    parser.add_argument("--move-clean", action="store_true", help="Also move particles that already pass clean field support.")
    args = parser.parse_args(argv)
    report = offset_boundary_release_particles(
        args.config,
        args.out_csv,
        particles_csv=args.particles_csv,
        report_json=args.report_json,
        offset_cells=float(args.offset_cells),
        offset_m=args.offset_m,
        max_normal_offset_cells=float(args.max_normal_offset_cells),
        max_tangent_offset_cells=float(args.max_tangent_offset_cells),
        preserve_clean=not bool(args.move_clean),
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
