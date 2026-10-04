from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from ._common import (
    _find_column,
    _numeric_column,
    _text_column,
    finite_summary,
    read_csv,
    write_csv,
)

ALIASES = {
    "particle_id": ("particle_id", "ParticleID", "id", "pid", "particle"),
    "hit_time_s": ("hit_time_s", "time_s", "time", "t", "wall_time_s", "event_time_s"),
    "part_id": (
        "part_id",
        "solver_part_id",
        "boundary_part_id",
        "comsol_entity_id",
        "entity_id",
        "boundary_id",
    ),
    "outcome": ("outcome", "wall_outcome", "status", "state"),
    "hit_x": ("hit_x", "hit_x_m", "x", "x_m", "event_x"),
    "hit_y": ("hit_y", "hit_y_m", "y", "y_m", "event_y"),
    "hit_z": ("hit_z", "hit_z_m", "z", "z_m", "event_z"),
    "normal_x": ("normal_x", "normal_x_m", "nx", "n_x"),
    "normal_y": ("normal_y", "normal_y_m", "ny", "n_y"),
    "normal_z": ("normal_z", "normal_z_m", "nz", "n_z"),
    "v_hit_x": ("v_hit_x", "v_hit_x_mps", "vx", "v_x", "hit_vx"),
    "v_hit_y": ("v_hit_y", "v_hit_y_mps", "vy", "v_y", "hit_vy"),
    "v_hit_z": ("v_hit_z", "v_hit_z_mps", "vz", "v_z", "hit_vz"),
}


def _canonical_first_hits(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    particle_col = _find_column(frame, ALIASES["particle_id"])
    time_col = _find_column(frame, ALIASES["hit_time_s"])
    if particle_col is None or time_col is None:
        raise ValueError(
            f"{label} boundary CSV must contain particle_id and hit_time_s/time_s"
        )
    work = pd.DataFrame(
        {
            "particle_id": pd.to_numeric(frame[particle_col], errors="coerce"),
            "hit_time_s": pd.to_numeric(frame[time_col], errors="coerce"),
            "part_id": _numeric_column(frame, ALIASES["part_id"]),
            "outcome": _text_column(frame, ALIASES["outcome"]),
            "hit_x": _numeric_column(frame, ALIASES["hit_x"]),
            "hit_y": _numeric_column(frame, ALIASES["hit_y"]),
            "hit_z": _numeric_column(frame, ALIASES["hit_z"]),
            "normal_x": _numeric_column(frame, ALIASES["normal_x"]),
            "normal_y": _numeric_column(frame, ALIASES["normal_y"]),
            "normal_z": _numeric_column(frame, ALIASES["normal_z"]),
            "v_hit_x": _numeric_column(frame, ALIASES["v_hit_x"]),
            "v_hit_y": _numeric_column(frame, ALIASES["v_hit_y"]),
            "v_hit_z": _numeric_column(frame, ALIASES["v_hit_z"]),
        }
    )
    work = work[
        np.isfinite(work["particle_id"].to_numpy(dtype=float))
        & np.isfinite(work["hit_time_s"].to_numpy(dtype=float))
    ]
    if work.empty:
        return work
    work["particle_id"] = work["particle_id"].astype(int)
    work = work.sort_values(["particle_id", "hit_time_s"], kind="mergesort")
    first = work.groupby("particle_id", as_index=False).first()
    counts = (
        work.groupby("particle_id", as_index=False)
        .size()
        .rename(columns={"size": f"{label}_event_count"})
    )
    return first.merge(counts, on="particle_id", how="left")


def _norm_error(frame: pd.DataFrame, bases: tuple[str, ...]) -> np.ndarray:
    deltas = []
    for base in bases:
        left = f"{base}_python"
        right = f"{base}_comsol"
        if left in frame.columns and right in frame.columns:
            delta = pd.to_numeric(frame[left], errors="coerce") - pd.to_numeric(
                frame[right], errors="coerce"
            )
            if np.isfinite(delta.to_numpy(dtype=float)).any():
                deltas.append(delta.to_numpy(dtype=float))
    if not deltas:
        return np.full(len(frame), np.nan, dtype=float)
    stack = np.vstack(deltas)
    valid = np.all(np.isfinite(stack), axis=0)
    out = np.full(len(frame), np.nan, dtype=float)
    out[valid] = np.sqrt(np.sum(stack[:, valid] ** 2, axis=0))
    return out


def _summary(merged: pd.DataFrame) -> dict[str, object]:
    both = merged["_merge"] == "both"
    out: dict[str, object] = {
        "particle_count": len(merged),
        "matched_first_hit_count": int(np.count_nonzero(both.to_numpy())),
        "python_only_count": int(
            np.count_nonzero((merged["_merge"] == "left_only").to_numpy())
        ),
        "comsol_only_count": int(
            np.count_nonzero((merged["_merge"] == "right_only").to_numpy())
        ),
    }
    for name in (
        "hit_time_error_s",
        "hit_position_error_m",
        "normal_error",
        "hit_velocity_error_mps",
    ):
        if name not in merged.columns:
            continue
        values = pd.to_numeric(merged.loc[both, name], errors="coerce").to_numpy(
            dtype=float
        )
        out[name] = finite_summary(values)
    for base in ("part_id", "outcome"):
        col = f"{base}_match"
        if col in merged.columns:
            values = merged.loc[both, col].dropna().astype(bool)
            out[f"{base}_match_ratio"] = float(values.mean()) if len(values) else None
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="particle-tracer compare boundary",
        description=(
            "Compare first wall-hit diagnostics against COMSOL boundary-hit CSV."
        ),
    )
    parser.add_argument("--python", dest="python_csv", required=True, type=Path)
    parser.add_argument("--comsol", required=True, type=Path)
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=Path("boundary_hit_comparison.csv"),
    )
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args(argv)

    python_df = _canonical_first_hits(read_csv(args.python_csv), label="python")
    comsol_df = _canonical_first_hits(read_csv(args.comsol), label="comsol")
    merged = python_df.merge(
        comsol_df,
        on=["particle_id"],
        how="outer",
        suffixes=("_python", "_comsol"),
        indicator=True,
    )
    merged["hit_time_error_s"] = (
        pd.to_numeric(merged["hit_time_s_python"], errors="coerce")
        - pd.to_numeric(merged["hit_time_s_comsol"], errors="coerce")
    ).abs()
    for base in ("part_id", "outcome"):
        left = f"{base}_python"
        right = f"{base}_comsol"
        if left in merged.columns and right in merged.columns:
            merged[f"{base}_match"] = merged[left].astype(str) == merged[right].astype(
                str
            )
    merged["hit_position_error_m"] = _norm_error(merged, ("hit_x", "hit_y", "hit_z"))
    merged["normal_error"] = _norm_error(merged, ("normal_x", "normal_y", "normal_z"))
    merged["hit_velocity_error_mps"] = _norm_error(
        merged, ("v_hit_x", "v_hit_y", "v_hit_z")
    )
    write_csv(merged, args.output)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(_summary(merged), indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
