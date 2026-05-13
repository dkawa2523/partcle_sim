from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ._common import read_csv, write_csv


def _merge_keys(left_columns: set[str], right_columns: set[str]) -> list[str]:
    keys = ["particle_id"]
    if "time" in left_columns and "time" in right_columns:
        keys.append("time")
    elif "t" in left_columns and "t" in right_columns:
        keys.append("t")
    elif "sample_index" in left_columns and "sample_index" in right_columns:
        keys.append("sample_index")
    return keys


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Python and COMSOL trajectory CSV files.")
    parser.add_argument("--python", dest="python_csv", required=True, type=Path)
    parser.add_argument("--comsol", required=True, type=Path)
    parser.add_argument("--output", "--out", dest="output", type=Path, default=Path("trajectory_error.csv"))
    args = parser.parse_args()

    python_df = read_csv(args.python_csv)
    comsol_df = read_csv(args.comsol)
    keys = _merge_keys(set(python_df.columns), set(comsol_df.columns))
    merged = python_df.merge(comsol_df, on=keys, how="outer", suffixes=("_python", "_comsol"), indicator=True)
    components = [name for name in ("x", "y", "z") if f"{name}_python" in merged.columns and f"{name}_comsol" in merged.columns]
    for name in components:
        merged[f"d{name}"] = merged[f"{name}_python"].astype(float) - merged[f"{name}_comsol"].astype(float)
    if components:
        merged["position_error"] = np.sqrt(sum(merged[f"d{name}"].astype(float) ** 2 for name in components))
    write_csv(merged, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
