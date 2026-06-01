from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def first_present(row: Mapping[str, object], names: Sequence[str], default: object = np.nan) -> object:
    for name in names:
        if name in row and pd.notna(row[name]):
            value = row[name]
            if not (isinstance(value, str) and value.strip() == ""):
                return value
    return default


def row_time(row: Mapping[str, object]) -> float:
    return float(first_present(row, ("time", "t", "time_s", "release_time"), 0.0))


def row_point_id(row: Mapping[str, object], fallback: int) -> object:
    return first_present(row, ("point_id", "sample_id", "id"), fallback)


def row_particle_id(row: Mapping[str, object]) -> int | None:
    value = first_present(row, ("particle_id", "pid", "id"), None)
    if value is None or pd.isna(value):
        return None
    return int(value)


def row_position(row: Mapping[str, object], spatial_dim: int) -> np.ndarray:
    names = (("x", "r", "position_x"), ("y", "z", "position_y"), ("z", "position_z"))
    values = []
    for axis in range(int(spatial_dim)):
        values.append(float(first_present(row, names[axis], 0.0)))
    return np.asarray(values, dtype=np.float64)


def row_velocity(row: Mapping[str, object], spatial_dim: int) -> np.ndarray | None:
    names = (("vx", "vr", "velocity_x"), ("vy", "vz", "velocity_y"), ("vz", "velocity_z"))
    if not any(any(name in row for name in axis_names) for axis_names in names[: int(spatial_dim)]):
        return None
    values = []
    for axis in range(int(spatial_dim)):
        values.append(float(first_present(row, names[axis], 0.0)))
    return np.asarray(values, dtype=np.float64)


def relative_error(python_value: float, reference_value: float) -> float:
    denom = max(abs(float(reference_value)), 1.0e-30)
    return float(abs(float(python_value) - float(reference_value)) / denom)


def component_labels(spatial_dim: int) -> tuple[str, ...]:
    return ("x", "y", "z")[: int(spatial_dim)]


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def as_long_reference(df: pd.DataFrame, *, value_name: str = "comsol_value") -> pd.DataFrame:
    columns = set(df.columns)
    value_col = next((name for name in ("comsol_value", "reference_value", "value") if name in columns), None)
    if {"point_id", "component"} <= columns and value_col is not None:
        out = df.copy()
        if value_col != value_name:
            out = out.rename(columns={value_col: value_name})
        return out
    id_col = "point_id" if "point_id" in columns else ("sample_id" if "sample_id" in columns else None)
    if id_col is None:
        raise ValueError("reference CSV must contain point_id/sample_id for wide or long comparison")
    value_columns = [name for name in df.columns if name not in {id_col, "time", "t", "time_s"}]
    rows = []
    for _, row in df.iterrows():
        for name in value_columns:
            rows.append(
                {
                    "point_id": row[id_col],
                    "component": str(name),
                    value_name: row[name],
                }
            )
    return pd.DataFrame(rows)


def merge_with_reference(sampled: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    ref = as_long_reference(reference)
    keys = ["point_id", "component"]
    if "field" in sampled.columns and "field" in ref.columns:
        keys = ["point_id", "field", "component"]
    merged = sampled.merge(ref, on=keys, how="left")
    if "comsol_value" in merged.columns:
        merged["abs_error"] = (merged["python_value"].astype(float) - merged["comsol_value"].astype(float)).abs()
        merged["rel_error"] = [
            relative_error(py, refv)
            for py, refv in zip(merged["python_value"].to_numpy(), merged["comsol_value"].to_numpy())
        ]
        merged["sign_match"] = np.sign(merged["python_value"].astype(float)) == np.sign(merged["comsol_value"].astype(float))
    return merged


def normalize_components(values: Iterable[object]) -> list[str]:
    out = []
    for value in values:
        text = str(value).strip()
        if text:
            out.append(text)
    return out
