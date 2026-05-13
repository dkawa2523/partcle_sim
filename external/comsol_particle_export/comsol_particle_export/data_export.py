from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


_COLUMN_RE = re.compile(
    r"^\s*(?P<name>.+?)(?:\s+\((?P<unit>[^()]*)\))?\s+@\s*t\s*=\s*(?P<time>[-+0-9.eE]+)\s*$"
)

_UNIT_SCALE_TO_M = {
    "m": 1.0,
    "meter": 1.0,
    "meters": 1.0,
    "mm": 1.0e-3,
    "millimeter": 1.0e-3,
    "millimeters": 1.0e-3,
    "um": 1.0e-6,
    "µm": 1.0e-6,
    "micrometer": 1.0e-6,
    "micrometers": 1.0e-6,
    "nm": 1.0e-9,
}


def _read_rows(path: Path) -> tuple[dict[str, str], list[str], list[list[str]]]:
    csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
    metadata: dict[str, str] = {}
    header: list[str] | None = None
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("%"):
                body = line[1:].lstrip()
                parsed = next(csv.reader([body]))
                if parsed and parsed[0].strip().lower() == "index":
                    header = parsed
                elif len(parsed) >= 2:
                    metadata[parsed[0].strip()] = ",".join(parsed[1:]).strip()
                continue
            rows.append(next(csv.reader([line])))
    if header is None:
        raise ValueError(f"COMSOL Data export is missing a '% Index,...' header: {path}")
    return metadata, header, rows


def _parse_columns(header: Sequence[str], axis_names: Iterable[str]) -> list[dict[str, Any]]:
    wanted = {str(name).strip().lower(): str(name).strip() for name in axis_names}
    parsed: list[dict[str, Any]] = []
    for index, label in enumerate(header[1:], start=1):
        match = _COLUMN_RE.match(str(label))
        if not match:
            continue
        raw_name = match.group("name").strip()
        axis = wanted.get(raw_name.lower())
        if axis is None:
            continue
        parsed.append(
            {
                "column_index": index,
                "axis": axis,
                "raw_name": raw_name,
                "unit": (match.group("unit") or "").strip(),
                "time_s": float(match.group("time")),
            }
        )
    if not parsed:
        raise ValueError("COMSOL Data export contains no requested axis columns")
    return parsed


def _parse_mapped_columns(header: Sequence[str], expression_map: dict[str, str]) -> list[dict[str, Any]]:
    wanted = {str(raw).strip().lower(): str(out).strip() for raw, out in expression_map.items()}
    parsed: list[dict[str, Any]] = []
    for index, label in enumerate(header[1:], start=1):
        match = _COLUMN_RE.match(str(label))
        if not match:
            continue
        raw_name = match.group("name").strip()
        output_name = wanted.get(raw_name.lower())
        if output_name is None:
            continue
        parsed.append(
            {
                "column_index": index,
                "output_name": output_name,
                "raw_name": raw_name,
                "unit": (match.group("unit") or "").strip(),
                "time_s": float(match.group("time")),
            }
        )
    if not parsed:
        raise ValueError("COMSOL Data export contains no requested expression columns")
    return parsed


def _unit_scale_to_m(unit: str, fallback: float) -> float:
    key = str(unit).strip().lower().replace("μ", "µ")
    if not key:
        return float(fallback)
    return _UNIT_SCALE_TO_M.get(key, float(fallback))


def _finite(value: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def canonicalize_particle_wide_data_export(
    data_export_csv: str | Path,
    *,
    expression_map: dict[str, str],
    required_output_columns: Sequence[str] | None = None,
    fallback_unit_scale: float = 1.0,
    drop_incomplete: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert a COMSOL Data export wide table into canonical long form.

    `expression_map` maps raw COMSOL expression labels such as `x` or `fpt.vx`
    to output column names such as `x` or `v_x`.  The table is assumed to use
    the common COMSOL Particle dataset shape: one row per particle node and one
    column per expression/time pair.
    """

    path = Path(data_export_csv)
    if not expression_map:
        raise ValueError("expression_map must not be empty")
    output_names = list(dict.fromkeys(str(v).strip() for v in expression_map.values()))
    required = [str(name) for name in (required_output_columns or output_names)]
    metadata, header, raw_rows = _read_rows(path)
    parsed = _parse_mapped_columns(header, expression_map)
    by_time: dict[float, dict[str, dict[str, Any]]] = {}
    units_by_output: dict[str, set[str]] = {name: set() for name in output_names}
    for col in parsed:
        by_time.setdefault(float(col["time_s"]), {})[str(col["output_name"])] = col
        units_by_output[str(col["output_name"])].add(str(col["unit"]))

    complete_times = [
        time for time, columns in sorted(by_time.items())
        if all(name in columns for name in required)
    ]
    if not complete_times:
        raise ValueError(f"COMSOL Data export has no time with complete required columns: {', '.join(required)}")

    scale_by_output = {}
    for name in output_names:
        units = units_by_output[name]
        unit = next(iter(units - {""}), "")
        scale_by_output[name] = _unit_scale_to_m(unit, fallback_unit_scale)

    out_rows: list[dict[str, float | int]] = []
    particle_ids: set[int] = set()
    for row in raw_rows:
        if not row:
            continue
        try:
            particle_id = int(float(row[0]))
        except ValueError:
            continue
        particle_ids.add(particle_id)
        for time in complete_times:
            record: dict[str, float | int] = {"particle_id": particle_id, "time_s": float(time)}
            required_finite_count = 0
            any_finite = False
            for name in output_names:
                col = by_time[time].get(name)
                value = math.nan
                if col is not None:
                    index = int(col["column_index"])
                    value = _finite(row[index]) if index < len(row) else math.nan
                if math.isfinite(value):
                    any_finite = True
                    value *= scale_by_output[name]
                    if name in required:
                        required_finite_count += 1
                record[name] = value
            if drop_incomplete and required_finite_count != len(required):
                continue
            if any_finite:
                out_rows.append(record)

    frame = pd.DataFrame(out_rows, columns=["particle_id", "time_s", *output_names])
    if not frame.empty:
        frame = frame.sort_values(["particle_id", "time_s"], kind="mergesort").reset_index(drop=True)

    finite_times = frame["time_s"] if "time_s" in frame.columns and not frame.empty else pd.Series(dtype=float)
    report = {
        "source_kind": "external_comsol_particle_export_canonical_wide_table",
        "input_csv": str(path),
        "metadata": metadata,
        "expression_map": dict(expression_map),
        "output_columns": output_names,
        "required_output_columns": required,
        "column_units": {name: sorted(units_by_output[name]) for name in output_names},
        "column_scale_to_target": scale_by_output,
        "raw_particle_count": int(len(particle_ids)),
        "raw_time_count": int(len(complete_times)),
        "trajectory_row_count": int(len(frame)),
        "trajectory_particle_count": int(frame["particle_id"].nunique()) if not frame.empty else 0,
        "trajectory_time_count": int(frame["time_s"].nunique()) if not frame.empty else 0,
        "time_min_s": float(finite_times.min()) if not finite_times.empty else None,
        "time_max_s": float(finite_times.max()) if not finite_times.empty else None,
        "dropped_incomplete_samples": bool(drop_incomplete),
    }
    return frame, report


def canonicalize_particle_xy_data_export(
    data_export_csv: str | Path,
    *,
    axis_names: Sequence[str] = ("x", "y"),
    fallback_coordinate_scale_m_per_unit: float = 1.0,
    drop_incomplete: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert COMSOL Data export wide particle coordinates to long trajectories.

    COMSOL's Data export for a Particle dataset is typically row-oriented by
    particle node and column-oriented by expression/time, for example
    `x (mm) @ t=0,y (mm) @ t=0,...`.  The returned frame is canonical long
    form with `particle_id`, `time_s`, and requested axis columns scaled to m.
    """

    axes = [str(name) for name in axis_names]
    frame, report = canonicalize_particle_wide_data_export(
        data_export_csv,
        expression_map={axis: axis for axis in axes},
        required_output_columns=axes,
        fallback_unit_scale=fallback_coordinate_scale_m_per_unit,
        drop_incomplete=drop_incomplete,
    )
    report.update({
        "source_kind": "external_comsol_particle_export_canonical_trajectory",
        "axis_names": axes,
        "axis_units": {axis: report["column_units"][axis] for axis in axes},
        "axis_scale_to_m": {axis: report["column_scale_to_target"][axis] for axis in axes},
    })
    return frame, report


def write_canonical_particle_trajectory(
    data_export_csv: str | Path,
    out_csv: str | Path,
    *,
    axis_names: Sequence[str] = ("x", "y"),
    fallback_coordinate_scale_m_per_unit: float = 1.0,
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    frame, report = canonicalize_particle_xy_data_export(
        data_export_csv,
        axis_names=axis_names,
        fallback_coordinate_scale_m_per_unit=fallback_coordinate_scale_m_per_unit,
    )
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    report["output_csv"] = str(out)
    if report_json is not None:
        report_path = Path(report_json)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def derive_particle_tables_from_trajectory(
    trajectory_csv: str | Path,
    *,
    release_csv: str | Path | None = None,
    final_csv: str | Path | None = None,
    initial_velocity: dict[str, float] | None = None,
    final_state: str = "unknown",
    report_json: str | Path | None = None,
) -> dict[str, Any]:
    """Write first-sample release and last-sample final tables from trajectory CSV."""

    path = Path(trajectory_csv)
    frame = pd.read_csv(path)
    if "particle_id" not in frame.columns or "time_s" not in frame.columns:
        raise ValueError("trajectory CSV must contain particle_id and time_s")
    if frame.empty:
        raise ValueError("trajectory CSV is empty")
    work = frame.sort_values(["particle_id", "time_s"], kind="mergesort")
    first = work.groupby("particle_id", as_index=False).first()
    last = work.groupby("particle_id", as_index=False).last()
    position_cols = [col for col in ("x", "y", "z", "r") if col in work.columns]
    velocity_cols = [col for col in ("v_x", "v_y", "v_z", "vx", "vy", "vz") if col in work.columns]

    release = first[["particle_id", "time_s", *position_cols]].rename(columns={"time_s": "release_time"})
    velocity_source = "first_trajectory_sample"
    if initial_velocity is not None:
        velocity_source = "override"
        for col, value in initial_velocity.items():
            release[str(col)] = float(value)
    else:
        for col in velocity_cols:
            release[col] = first[col]

    final_cols = ["particle_id", "time_s", *position_cols, *velocity_cols]
    final = last[[col for col in final_cols if col in last.columns]].copy()
    final["final_state"] = str(final_state)

    report = {
        "source_kind": "external_comsol_particle_export_derived_particle_tables",
        "trajectory_csv": str(path),
        "particle_count": int(work["particle_id"].nunique()),
        "trajectory_row_count": int(len(work)),
        "release_time_min_s": float(release["release_time"].min()),
        "release_time_max_s": float(release["release_time"].max()),
        "final_time_min_s": float(final["time_s"].min()),
        "final_time_max_s": float(final["time_s"].max()),
        "position_columns": position_cols,
        "velocity_columns": velocity_cols,
        "release_velocity_source": velocity_source,
    }
    if release_csv is not None:
        out = Path(release_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        release.to_csv(out, index=False)
        report["release_csv"] = str(out)
    if final_csv is not None:
        out = Path(final_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(out, index=False)
        report["final_csv"] = str(out)
    if report_json is not None:
        out = Path(report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
