from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from particle_tracer_unified.compare._common import read_csv, write_csv

_TIME_COLUMNS = ("time_s", "time", "t")


def _normalize_trajectory_key(
    frame: pd.DataFrame,
    option: str,
) -> tuple[pd.DataFrame, str]:
    time_columns = [name for name in _TIME_COLUMNS if name in frame.columns]
    if len(time_columns) > 1:
        raise ValueError(
            f"{option} trajectory CSV contains multiple time columns: "
            f"{', '.join(time_columns)}"
        )
    if time_columns:
        time_column = time_columns[0]
        if time_column != "time_s":
            frame = frame.rename(columns={time_column: "time_s"})
        return frame, "time_s"
    if "sample_index" in frame.columns:
        return frame, "sample_index"
    raise ValueError(
        f"{option} trajectory CSV must contain one of time_s/time/t or sample_index"
    )


def _require_finite_column(
    frame: pd.DataFrame,
    column: str,
    option: str,
) -> None:
    try:
        values = np.asarray(
            pd.to_numeric(frame[column], errors="raise"),
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{option} trajectory CSV contains non-numeric {column}"
        ) from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{option} trajectory CSV contains non-finite {column}")


def _validate_trajectory_keys(
    frame: pd.DataFrame,
    option: str,
    sample_key: str,
) -> None:
    keys = ["particle_id", sample_key]
    for key in keys:
        _require_finite_column(frame, key, option)
    if bool(frame.duplicated(keys).any()):
        raise ValueError(
            f"{option} trajectory CSV contains duplicate trajectory key "
            f"({', '.join(keys)})"
        )


def _read_trajectory_csv(path: Path, option: str) -> tuple[pd.DataFrame, str]:
    if path.suffix.lower() in {".npy", ".npz"}:
        raise ValueError(
            f"{option} expects a long-form trajectory CSV, not {path.name}"
        )
    frame = read_csv(path)
    if "particle_id" not in frame.columns:
        raise ValueError(f"{option} trajectory CSV must contain particle_id")
    frame, sample_key = _normalize_trajectory_key(frame, option)
    _validate_trajectory_keys(frame, option, sample_key)
    return frame, sample_key


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="particle-tracer compare trajectory",
        description="Compare Python and COMSOL trajectory CSV files.",
    )
    parser.add_argument(
        "--python",
        dest="python_csv",
        required=True,
        type=Path,
        help=(
            "Long-form Python trajectory CSV with particle_id and x/y[/z]; "
            "debug trajectory.npy and trajectory_frames.csv are not trajectory tables"
        ),
    )
    parser.add_argument(
        "--comsol",
        required=True,
        type=Path,
        help="Long-form COMSOL trajectory CSV with particle_id and x/y[/z]",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=Path("trajectory_error.csv"),
    )
    return parser


def _shared_components(
    python_df: pd.DataFrame,
    comsol_df: pd.DataFrame,
) -> list[str]:
    python_columns = set(python_df.columns)
    comsol_columns = set(comsol_df.columns)
    components = [
        name for name in ("x", "y", "z") if name in python_columns & comsol_columns
    ]
    if not components:
        raise ValueError("trajectory CSVs must share at least one x/y/z column")
    for option, frame in (("--python", python_df), ("--comsol", comsol_df)):
        for component in components:
            _require_finite_column(frame, component, option)
    return components


def _compare_trajectories(
    python_df: pd.DataFrame,
    python_sample_key: str,
    comsol_df: pd.DataFrame,
    comsol_sample_key: str,
) -> pd.DataFrame:
    if python_sample_key != comsol_sample_key:
        raise ValueError(
            "trajectory CSVs must use the same sample key: time_s or sample_index"
        )
    components = _shared_components(python_df, comsol_df)
    keys = ["particle_id", python_sample_key]
    merged = python_df.merge(
        comsol_df,
        on=keys,
        how="outer",
        suffixes=("_python", "_comsol"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = merged["_merge"] != "both"
    if bool(unmatched.any()):
        raise ValueError(
            "trajectory keys do not match exactly: "
            f"{int(np.count_nonzero(unmatched))} unmatched row(s)"
        )
    for name in components:
        merged[f"d{name}"] = merged[f"{name}_python"].astype(float) - merged[
            f"{name}_comsol"
        ].astype(float)
    squared_errors = [merged[f"d{name}"].astype(float) ** 2 for name in components]
    with np.errstate(over="ignore", invalid="ignore"):
        merged["position_error"] = np.sqrt(sum(squared_errors))
    if not np.all(np.isfinite(merged["position_error"].to_numpy(dtype=np.float64))):
        raise ValueError("trajectory comparison produced non-finite position_error")
    return merged


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)

    python_df, python_sample_key = _read_trajectory_csv(args.python_csv, "--python")
    comsol_df, comsol_sample_key = _read_trajectory_csv(args.comsol, "--comsol")
    merged = _compare_trajectories(
        python_df,
        python_sample_key,
        comsol_df,
        comsol_sample_key,
    )
    write_csv(merged, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
