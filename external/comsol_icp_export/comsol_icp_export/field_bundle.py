from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

AXIS_COLUMNS = ("r", "z")
REQUIRED_SAMPLE_COLUMNS = ("ux", "uy", "mu", "E_x", "E_y")
RESERVED_COLUMNS = set(AXIS_COLUMNS) | {"valid_mask"}


def load_json_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON mapping expected: {path}")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(payload), indent=2), encoding="utf-8")


def _as_table(table: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table.copy()
    return pd.read_csv(Path(table))


def _numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise ValueError(f"missing required field sample column: {name}")
    values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
    return values


def _axis_from_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    values = _numeric_column(frame, name)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} axis contains non-finite values")
    axis = np.asarray(sorted(np.unique(values)), dtype=np.float64)
    if axis.size < 2:
        raise ValueError(f"{name} axis must contain at least two unique points")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} axis must be strictly increasing")
    return axis


def _grid_from_column(frame: pd.DataFrame, axis_r: np.ndarray, axis_z: np.ndarray, name: str) -> np.ndarray:
    values = _numeric_column(frame, name)
    work = frame[["r", "z"]].copy()
    work[name] = values
    pivot = work.pivot(index="r", columns="z", values=name)
    grid = pivot.reindex(index=axis_r, columns=axis_z).to_numpy(dtype=np.float64)
    return grid


def _variation(values: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return 0.0
    return float(np.max(selected) - np.min(selected))


def _is_nonuniform(values: np.ndarray, mask: np.ndarray, *, atol: float = 1.0e-30, rtol: float = 1.0e-9) -> bool:
    selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    selected = selected[np.isfinite(selected)]
    if selected.size < 2:
        return False
    span = float(np.max(selected) - np.min(selected))
    scale = max(float(np.max(np.abs(selected))), 1.0)
    return span > max(float(atol), float(rtol) * scale)


def _masked_for_provider(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    out[~np.asarray(valid_mask, dtype=bool)] = np.nan
    return out


def _quantity_summary(arrays: Mapping[str, np.ndarray], valid_mask: np.ndarray) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    mask = np.asarray(valid_mask, dtype=bool)
    for key, values in arrays.items():
        arr = np.asarray(values, dtype=np.float64)
        selected = arr[mask]
        finite = selected[np.isfinite(selected)]
        if finite.size == 0:
            summary[key] = {"finite_count": 0, "variation": 0.0}
            continue
        summary[key] = {
            "finite_count": int(finite.size),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "variation": float(np.max(finite) - np.min(finite)),
        }
    return summary


def build_field_bundle_from_table(
    table: str | Path | pd.DataFrame,
    *,
    q_ref_c: float,
    m_ref_kg: float,
    coordinate_scale_m_per_model_unit: float = 1.0,
    coordinate_model_unit: str = "model_unit",
    metadata: Mapping[str, Any] | None = None,
    require_nonuniform: bool = True,
) -> dict[str, np.ndarray]:
    """Build a solver-compatible field bundle from COMSOL point samples.

    The input table must be a complete tensor grid in `(r, z)`.
    Required fields must be finite on `valid_mask`. Values outside
    `valid_mask` are stored as NaN so downstream provider support cannot
    silently expand beyond the exported field support.
    """

    frame = _as_table(table)
    missing = [c for c in (*AXIS_COLUMNS, *REQUIRED_SAMPLE_COLUMNS) if c not in frame.columns]
    if missing:
        raise ValueError(f"missing required field sample columns: {', '.join(missing)}")
    if frame.duplicated(["r", "z"]).any():
        raise ValueError("field samples contain duplicate r,z points")

    scale = float(coordinate_scale_m_per_model_unit)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("coordinate_scale_m_per_model_unit must be a positive finite value")
    raw_axis_r = _axis_from_column(frame, "r")
    raw_axis_z = _axis_from_column(frame, "z")
    axis_r = raw_axis_r * scale
    axis_z = raw_axis_z * scale
    expected_count = int(raw_axis_r.size * raw_axis_z.size)
    if int(len(frame)) != expected_count:
        raise ValueError(
            "field samples must form a complete tensor grid; "
            f"got {len(frame)} rows, expected {expected_count}"
        )

    grids = {name: _grid_from_column(frame, raw_axis_r, raw_axis_z, name) for name in REQUIRED_SAMPLE_COLUMNS}
    if "valid_mask" in frame.columns:
        valid_values = pd.to_numeric(frame["valid_mask"], errors="coerce").to_numpy(dtype=np.float64)
        valid_frame = frame[["r", "z"]].copy()
        valid_frame["valid_mask"] = valid_values
        valid_mask = (
            valid_frame.pivot(index="r", columns="z", values="valid_mask")
            .reindex(index=raw_axis_r, columns=raw_axis_z)
            .to_numpy(dtype=np.float64)
            > 0.5
        )
    else:
        valid_mask = np.ones((axis_r.size, axis_z.size), dtype=bool)
        for values in grids.values():
            valid_mask &= np.isfinite(values)

    if not np.any(valid_mask):
        raise ValueError("valid_mask contains no valid field-support nodes")
    for name, values in grids.items():
        if np.any(~np.isfinite(values[valid_mask])):
            raise ValueError(f"required field {name} is non-finite on valid support")

    if require_nonuniform:
        flow_nonuniform = _is_nonuniform(grids["ux"], valid_mask) or _is_nonuniform(grids["uy"], valid_mask)
        electric_nonuniform = _is_nonuniform(grids["E_x"], valid_mask) or _is_nonuniform(grids["E_y"], valid_mask)
        if not flow_nonuniform:
            raise ValueError("velocity field is spatially uniform on valid support")
        if not electric_nonuniform:
            raise ValueError("electric field is spatially uniform on valid support")

    arrays: dict[str, np.ndarray] = {
        "ux": _masked_for_provider(grids["ux"], valid_mask),
        "uy": _masked_for_provider(grids["uy"], valid_mask),
        "mu": _masked_for_provider(grids["mu"], valid_mask),
        "E_x": _masked_for_provider(grids["E_x"], valid_mask),
        "E_y": _masked_for_provider(grids["E_y"], valid_mask),
    }

    skipped_optional: dict[str, str] = {}
    for name in frame.columns:
        if name in RESERVED_COLUMNS or name in REQUIRED_SAMPLE_COLUMNS:
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        if values.isna().all():
            skipped_optional[name] = "not_numeric"
            continue
        grid = _grid_from_column(frame, raw_axis_r, raw_axis_z, name)
        if np.any(~np.isfinite(grid[valid_mask])):
            skipped_optional[name] = "nonfinite_on_valid_support"
            continue
        arrays[name] = _masked_for_provider(grid, valid_mask)

    meta = dict(metadata or {})
    meta.update(
        {
            "source_kind": "external_comsol_icp_export",
            "field_layout": "axis_0=r_m, axis_1=z_m",
            "raw_coordinate_model_unit": str(coordinate_model_unit),
            "coordinate_scale_m_per_model_unit": float(scale),
            "raw_axis_0_bounds_model_units": [float(raw_axis_r[0]), float(raw_axis_r[-1])],
            "raw_axis_1_bounds_model_units": [float(raw_axis_z[0]), float(raw_axis_z[-1])],
            "axis_0_bounds_m": [float(axis_r[0]), float(axis_r[-1])],
            "axis_1_bounds_m": [float(axis_z[0]), float(axis_z[-1])],
            "times": [0.0],
            "reference_particle_charge_C": float(q_ref_c),
            "reference_particle_mass_kg": float(m_ref_kg),
            "grid_shape": [int(axis_r.size), int(axis_z.size)],
            "valid_node_count": int(np.count_nonzero(valid_mask)),
            "required_columns": list(REQUIRED_SAMPLE_COLUMNS),
            "skipped_optional_columns": skipped_optional,
            "quantity_summary": _quantity_summary(arrays, valid_mask),
        }
    )

    bundle: dict[str, np.ndarray] = {
        "axis_0": axis_r.astype(np.float64),
        "axis_1": axis_z.astype(np.float64),
        "times": np.asarray([0.0], dtype=np.float64),
        "valid_mask": valid_mask.astype(bool),
        "metadata_json": np.asarray(json.dumps(meta)),
    }
    bundle.update(arrays)
    return bundle


def write_field_bundle(bundle: Mapping[str, np.ndarray], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{key: np.asarray(value) for key, value in bundle.items()})
