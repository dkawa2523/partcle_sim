from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _finite_stats(values: pd.Series) -> dict[str, Any]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
    }


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    counts = frame[column].value_counts(dropna=False).sort_index()
    out: dict[str, int] = {}
    for key, value in counts.items():
        if pd.isna(key):
            label = "nan"
        else:
            try:
                label = str(int(key))
            except (TypeError, ValueError):
                label = str(key)
        out[label] = int(value)
    return out


def summarize_input_contract(report_json: str | Path, violations_csv: str | Path | None = None) -> dict[str, Any]:
    report_path = Path(report_json)
    report = _read_json(report_path)
    if violations_csv is None:
        candidate = report_path.parent / "input_particle_violations.csv"
        violations_path = candidate if candidate.exists() else None
    else:
        violations_path = Path(violations_csv)

    summary: dict[str, Any] = {
        "source_report": str(report_path),
        "mode": report.get("mode"),
        "passed": bool(report.get("passed", False)),
        "particle_count": int(report.get("particle_count", 0)),
        "status_counts": report.get("status_counts", {}),
        "checked_time_min_s": report.get("checked_time_min_s"),
        "checked_time_max_s": report.get("checked_time_max_s"),
        "near_boundary_threshold_m": report.get("near_boundary_threshold_m"),
        "non_clean_near_boundary_count": int(report.get("non_clean_near_boundary_count", 0)),
        "non_clean_geometry_inside_count": int(report.get("non_clean_geometry_inside_count", 0)),
        "violations_csv": str(violations_path) if violations_path is not None else "",
    }

    if violations_path is None or not violations_path.exists():
        summary["violation_rows"] = 0
        return summary

    violations = pd.read_csv(violations_path)
    summary.update(
        {
            "violation_rows": int(len(violations)),
            "violation_status_counts": _value_counts(violations, "status"),
            "violation_source_part_counts": _value_counts(violations, "source_part_id"),
            "violation_nearest_boundary_part_counts": _value_counts(violations, "nearest_boundary_part_id"),
            "violation_release_time_s": _finite_stats(violations["release_time_s"]) if "release_time_s" in violations else {"count": 0},
            "violation_boundary_distance_m": (
                _finite_stats(violations["nearest_boundary_distance_m"])
                if "nearest_boundary_distance_m" in violations
                else {"count": 0}
            ),
            "near_boundary_by_cell_diagonal_count": (
                int(pd.to_numeric(violations["near_boundary_by_cell_diagonal"], errors="coerce").fillna(0).sum())
                if "near_boundary_by_cell_diagonal" in violations
                else 0
            ),
            "geometry_inside_violation_count": (
                int(pd.to_numeric(violations["geometry_inside"], errors="coerce").fillna(0).sum())
                if "geometry_inside" in violations
                else 0
            ),
        }
    )
    return summary


def write_summary(summary: dict[str, Any], out_json: str | Path) -> None:
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a compact summary for a solver input contract report.")
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--violations-csv", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = summarize_input_contract(args.report_json, args.violations_csv)
    write_summary(summary, args.out_json)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
