from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd


def _json_scalar(value: Any, *, finite_only: bool) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return None if finite_only and not np.isfinite(number) else number
    if finite_only and isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def json_safe(value: Any) -> Any:
    """Convert values using the comparison-artifact JSON policy."""

    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return _json_scalar(value, finite_only=False)


def finite_json_safe(value: Any) -> Any:
    """Convert diagnostics to strict JSON, replacing NaN and infinity with null."""

    if isinstance(value, Mapping):
        return {str(key): finite_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json_safe(item) for item in value]
    return _json_scalar(value, finite_only=True)


def finite_summary(values: Any) -> dict[str, Any]:
    """Summarize finite numeric values using the comparison-report schema."""

    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
    }


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def _find_column(frame: pd.DataFrame | None, aliases: Sequence[str]) -> str | None:
    if frame is None:
        return None
    columns = {str(column).strip().lower(): str(column) for column in frame.columns}
    for alias in aliases:
        found = columns.get(str(alias).strip().lower())
        if found is not None:
            return found
    return None


def _numeric_column(
    frame: pd.DataFrame, aliases: Sequence[str], *, default: float = np.nan
) -> pd.Series:
    column = _find_column(frame, aliases)
    if column is None:
        return pd.Series(default, index=frame.index, dtype=float)
    return cast(pd.Series, pd.to_numeric(frame[column], errors="coerce"))


def _text_column(frame: pd.DataFrame, aliases: Sequence[str]) -> pd.Series:
    column = _find_column(frame, aliases)
    if column is None:
        return pd.Series("", index=frame.index, dtype=object)
    return frame[column].fillna("").astype(str)


def first_present(row: Any, names: Sequence[str], default: Any = np.nan) -> Any:
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
    names = (
        ("vx", "vr", "velocity_x"),
        ("vy", "vz", "velocity_y"),
        ("vz", "velocity_z"),
    )
    if not any(
        any(name in row for name in axis_names)
        for axis_names in names[: int(spatial_dim)]
    ):
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


def _reference_value_column(columns: set[str]) -> str | None:
    return next(
        (
            name
            for name in ("comsol_value", "reference_value", "value")
            if name in columns
        ),
        None,
    )


def _reference_id_column(columns: set[str]) -> str:
    if "point_id" in columns:
        return "point_id"
    if "sample_id" in columns:
        return "sample_id"
    raise ValueError(
        "reference CSV must contain point_id/sample_id for wide or long comparison"
    )


def _normalized_long_reference(
    df: pd.DataFrame,
    *,
    value_column: str,
    value_name: str,
) -> pd.DataFrame:
    out = _with_canonical_time(df)
    return (
        out
        if value_column == value_name
        else out.rename(columns={value_column: value_name})
    )


def _with_canonical_time(df: pd.DataFrame) -> pd.DataFrame:
    time_columns = [name for name in ("time_s", "time", "t") if name in df.columns]
    if len(time_columns) > 1:
        raise ValueError("reference data contains multiple time columns")
    out = df.copy()
    if time_columns and time_columns[0] != "time_s":
        out = out.rename(columns={time_columns[0]: "time_s"})
    return out


def _wide_reference_rows(
    df: pd.DataFrame,
    *,
    id_column: str,
    value_name: str,
) -> list[dict[str, object]]:
    normalized = _with_canonical_time(df)
    excluded = {id_column, "time_s"}
    value_columns = [name for name in normalized.columns if name not in excluded]
    return [
        {
            "point_id": row[id_column],
            **({"time_s": row["time_s"]} if "time_s" in normalized.columns else {}),
            "component": str(name),
            value_name: row[name],
        }
        for _, row in normalized.iterrows()
        for name in value_columns
    ]


def as_long_reference(
    df: pd.DataFrame, *, value_name: str = "comsol_value"
) -> pd.DataFrame:
    columns = set(df.columns)
    value_col = _reference_value_column(columns)
    if {"point_id", "component"} <= columns and value_col is not None:
        return _normalized_long_reference(
            df,
            value_column=value_col,
            value_name=value_name,
        )
    return pd.DataFrame(
        _wide_reference_rows(
            df,
            id_column=_reference_id_column(columns),
            value_name=value_name,
        )
    )


def merge_with_reference(
    sampled: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    ref = as_long_reference(reference)
    sampled = _with_canonical_time(sampled)
    keys = ["point_id", "component"]
    if "field" in sampled.columns and "field" in ref.columns:
        keys = ["point_id", "field", "component"]
    if "time_s" in ref.columns:
        if "time_s" not in sampled.columns:
            raise ValueError(
                "sampled data must contain time_s when reference data does"
            )
        keys.append("time_s")
    for name, frame in (("sampled", sampled), ("reference", ref)):
        if frame.duplicated(keys, keep=False).any():
            raise ValueError(f"{name} data contains duplicate comparison keys: {keys}")
    merged = sampled.merge(ref, on=keys, how="left", validate="one_to_one")
    if "comsol_value" in merged.columns:
        merged["abs_error"] = (
            merged["python_value"].astype(float) - merged["comsol_value"].astype(float)
        ).abs()
        merged["rel_error"] = [
            relative_error(py, refv)
            for py, refv in zip(
                merged["python_value"].to_numpy(),
                merged["comsol_value"].to_numpy(),
                strict=True,
            )
        ]
        merged["sign_match"] = np.sign(merged["python_value"].astype(float)) == np.sign(
            merged["comsol_value"].astype(float)
        )
    return merged
