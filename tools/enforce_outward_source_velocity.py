from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _summary(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50.0)),
        "p90": float(np.percentile(arr, 90.0)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _position_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    if {"x", "y", "source_x", "source_y", "vx", "vy"}.issubset(df.columns):
        pos_cols = ["x", "y"]
        src_cols = ["source_x", "source_y"]
        vel_cols = ["vx", "vy"]
        if {"z", "source_z", "vz"}.issubset(df.columns):
            pos_cols.append("z")
            src_cols.append("source_z")
            vel_cols.append("vz")
        return pos_cols, src_cols, vel_cols
    missing = sorted({"x", "y", "source_x", "source_y", "vx", "vy"} - set(df.columns))
    raise ValueError(f"particles CSV is missing source-surface columns: {', '.join(missing)}")


def enforce_outward_velocity(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    pos_cols, src_cols, vel_cols = _position_columns(df)
    out = df.copy()

    pos = out[pos_cols].to_numpy(dtype=np.float64)
    src = out[src_cols].to_numpy(dtype=np.float64)
    vel = np.array(out[vel_cols].to_numpy(dtype=np.float64), copy=True)

    normals = pos - src
    norm = np.linalg.norm(normals, axis=1)
    if np.any(~np.isfinite(norm)) or np.any(norm <= 0.0):
        bad = int(np.count_nonzero((~np.isfinite(norm)) | (norm <= 0.0)))
        raise ValueError(f"cannot infer outward source normals for {bad} particles")
    normals = normals / norm[:, None]

    normal_speed_before = np.sum(vel * normals, axis=1)
    inward = normal_speed_before < 0.0
    vel[inward] = vel[inward] - 2.0 * normal_speed_before[inward, None] * normals[inward]
    normal_speed_after = np.sum(vel * normals, axis=1)

    out.loc[:, vel_cols] = vel
    speed_before = np.linalg.norm(df[vel_cols].to_numpy(dtype=np.float64), axis=1)
    speed_after = np.linalg.norm(vel, axis=1)
    out["initial_speed_mps"] = speed_after

    summary = {
        "mode": "reflect_inward_normal_component",
        "particle_count": int(len(out)),
        "position_columns": pos_cols,
        "source_columns": src_cols,
        "velocity_columns": vel_cols,
        "corrected_particle_count": int(np.count_nonzero(inward)),
        "inward_count_before": int(np.count_nonzero(normal_speed_before < 0.0)),
        "inward_count_after": int(np.count_nonzero(normal_speed_after < -1.0e-12)),
        "normal_speed_before_mps": _summary(normal_speed_before),
        "normal_speed_after_mps": _summary(normal_speed_after),
        "speed_before_mps": _summary(speed_before),
        "speed_after_mps": _summary(speed_after),
        "note": "The source normal is inferred from release_position - source_position. Only inward normal components are reflected; speed magnitude is preserved.",
    }
    return out, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Keep explicit source particles moving into the computational domain.")
    ap.add_argument("particles_csv", type=Path)
    ap.add_argument("--summary-json", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.particles_csv)
    corrected, summary = enforce_outward_velocity(df)

    summary_path = args.summary_json or args.particles_csv.with_name("source_velocity_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.dry_run:
        corrected.to_csv(args.particles_csv, index=False)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
