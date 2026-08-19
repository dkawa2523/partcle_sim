from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from particle_tracer_unified.application import load_case
from particle_tracer_unified.compare._common import (
    as_long_reference,
    finite_float,
    merge_with_reference,
    read_csv,
    row_point_id,
    row_position,
    row_time,
    write_csv,
)
from particle_tracer_unified.core.datamodel import SolverContext
from particle_tracer_unified.core.field_backend import (
    VALID_MASK_QUANTITY,
    ProviderSamplingBackend,
)
from particle_tracer_unified.domain import FieldRequest, sample_one

_FIELD_COMPONENT_CANDIDATES = {
    ("u", "x"): ("ux", "u_x", "vx"),
    ("u", "y"): ("uy", "u_y", "vy"),
    ("u", "z"): ("uz", "u_z", "vz"),
    ("u", "r"): ("ur", "u_r", "ux"),
    ("E", "x"): ("E_x", "Ex", "electric_x", "electric_field_x"),
    ("E", "y"): ("E_y", "Ey", "electric_y", "electric_field_y"),
    ("E", "z"): ("E_z", "Ez", "electric_z", "electric_field_z"),
    ("E", "r"): ("E_r", "Er", "electric_r", "electric_field_r"),
}
_POSITION_COLUMNS = (
    ("x", "r", "position_x"),
    ("y", "z", "position_y"),
    ("z", "position_z"),
)


def _row_values(row: pd.Series) -> dict[str, object]:
    return {str(column): value for column, value in row.items()}


def _resolve_quantity(
    field_quantities: set[str], field_name: str, component: str
) -> str:
    field = field_name.strip()
    comp = component.strip()
    candidates = [comp, f"{field}_{comp}", f"{field}{comp}"]
    candidates.extend(_FIELD_COMPONENT_CANDIDATES.get((field, comp), ()))
    candidates.extend(_FIELD_COMPONENT_CANDIDATES.get((field.upper(), comp), ()))
    for candidate in candidates:
        if candidate in field_quantities:
            return candidate
    return comp


def _sample_quantity(
    backend: ProviderSamplingBackend,
    quantity: str,
    position: np.ndarray,
    time_s: float,
) -> dict[str, object]:
    fields = sample_one(
        backend,
        np.asarray(position, dtype=np.float64),
        time_s,
        FieldRequest((quantity, VALID_MASK_QUANTITY)),
    )
    status = int(np.asarray(fields.values[VALID_MASK_QUANTITY], dtype=np.uint8)[0])
    missing_quantities = fields.metadata.get("missing_quantities", ())
    missing = isinstance(missing_quantities, (list, tuple, set, frozenset)) and (
        quantity in missing_quantities
    )
    reasons = tuple(fields.metadata.get("valid_mask_reason", ("unknown",)))
    cell_ids = np.asarray(fields.metadata.get("cell_id", [-1]), dtype=np.int64)
    value = float(np.asarray(fields.values[quantity], dtype=np.float64)[0])
    return {
        "value": value,
        "provider_kind": str(fields.metadata.get("backend_kind", "")),
        "status": status,
        "reason": "missing_quantity" if missing else str(reasons[0]),
        "cell_id": int(cell_ids[0]),
        "valid": bool(fields.supported[0] and np.isfinite(value) and not missing),
    }


def _validate_points(points: pd.DataFrame, spatial_dim: int) -> None:
    if points.empty:
        raise ValueError("points CSV must contain at least one row")
    missing_axes = [
        axis
        for axis, aliases in enumerate(_POSITION_COLUMNS[:spatial_dim])
        if not any(alias in points.columns for alias in aliases)
    ]
    if missing_axes:
        axes = ", ".join(str(axis) for axis in missing_axes)
        raise ValueError(f"points CSV is missing coordinate axis {axes}")


def _sample_without_reference(
    context: SolverContext[Any, Any],
    points: pd.DataFrame,
    quantities: list[str],
) -> list[dict[str, object]]:
    backend = ProviderSamplingBackend(context.field_provider)
    rows: list[dict[str, object]] = []
    for fallback_idx, (_, row) in enumerate(points.iterrows()):
        values = _row_values(row)
        pos = row_position(values, context.spatial_dim)
        t_eval = row_time(values)
        point_id = row_point_id(values, fallback_idx)
        for quantity in quantities:
            sample = _sample_quantity(backend, quantity, pos, t_eval)
            rows.append(
                {
                    "point_id": point_id,
                    "time": t_eval,
                    "x": float(pos[0]),
                    "y": float(pos[1]) if context.spatial_dim >= 2 else np.nan,
                    "z": float(pos[2]) if context.spatial_dim == 3 else np.nan,
                    "field": "",
                    "component": quantity,
                    "quantity": quantity,
                    "python_value": float(sample["value"]),
                    "provider_kind": sample["provider_kind"],
                    "provider_status": int(sample["status"]),
                    "provider_reason": sample["reason"],
                    "cell_id": int(sample["cell_id"]),
                    "valid": int(bool(sample["valid"])),
                }
            )
    return rows


def _sample_reference_shape(
    context: SolverContext[Any, Any],
    points: pd.DataFrame,
    reference: pd.DataFrame,
) -> list[dict[str, object]]:
    backend = ProviderSamplingBackend(context.field_provider)
    quantities = set(context.field_provider.field.quantities.keys())
    points_by_id = _point_rows_by_id(points)
    rows: list[dict[str, object]] = []
    for fallback_idx, (_, ref_row) in enumerate(reference.iterrows()):
        reference_values = _row_values(ref_row)
        point_id = row_point_id(reference_values, fallback_idx)
        reference_time = _reference_time(reference_values)
        point_row = _reference_point_row(
            point_id=point_id,
            reference_time=reference_time,
            candidates=points_by_id.get(point_id, []),
            fallback=reference_values,
        )
        field_name = str(reference_values.get("field", ""))
        component = str(
            reference_values.get(
                "component",
                reference_values.get("quantity", ""),
            )
        ).strip()
        quantity = str(reference_values.get("quantity", "")).strip()
        if not quantity:
            quantity = _resolve_quantity(quantities, field_name, component)
        pos = row_position(point_row, context.spatial_dim)
        t_eval = reference_time if reference_time is not None else row_time(point_row)
        sample = _sample_quantity(backend, quantity, pos, t_eval)
        rows.append(
            {
                "point_id": point_id,
                "time": t_eval,
                "x": float(pos[0]),
                "y": float(pos[1]) if context.spatial_dim >= 2 else np.nan,
                "z": float(pos[2]) if context.spatial_dim == 3 else np.nan,
                "field": field_name,
                "component": component or quantity,
                "quantity": quantity,
                "python_value": float(sample["value"]),
                "provider_kind": sample["provider_kind"],
                "provider_status": int(sample["status"]),
                "provider_reason": sample["reason"],
                "cell_id": int(sample["cell_id"]),
                "valid": int(bool(sample["valid"])),
            }
        )
    return rows


def _point_rows_by_id(
    points: pd.DataFrame,
) -> dict[object, list[dict[str, object]]]:
    rows: dict[object, list[dict[str, object]]] = {}
    for fallback_idx, (_, row) in enumerate(points.iterrows()):
        values = _row_values(row)
        rows.setdefault(row_point_id(values, fallback_idx), []).append(values)
    return rows


def _reference_time(values: dict[str, object]) -> float | None:
    if "time_s" not in values:
        return None
    time_s = finite_float(values["time_s"])
    if not np.isfinite(time_s):
        raise ValueError("reference time_s must be numeric and finite")
    return time_s


def _reference_point_row(
    *,
    point_id: object,
    reference_time: float | None,
    candidates: list[dict[str, object]],
    fallback: dict[str, object],
) -> dict[str, object]:
    if reference_time is None:
        if len(candidates) > 1:
            raise ValueError(
                f"reference point_id {point_id!r} needs time_s because "
                "the points CSV contains multiple rows"
            )
        return candidates[0] if candidates else fallback
    matching = [row for row in candidates if row_time(row) == reference_time]
    if len(matching) != 1:
        raise ValueError(
            f"reference point_id {point_id!r} time_s {reference_time!r} "
            "must match exactly one points CSV row"
        )
    return matching[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="particle-tracer compare field",
        description="Compare sampled Python field values against COMSOL CSV samples.",
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="particle_tracer_unified run YAML",
    )
    parser.add_argument(
        "--points",
        required=True,
        type=Path,
        help="CSV with point_id,x,y[,z],time columns",
    )
    parser.add_argument(
        "--comsol", "--reference", dest="reference", type=Path, default=None
    )
    parser.add_argument(
        "--quantities",
        nargs="*",
        default=None,
        help="Field quantity names to sample when no reference is provided",
    )
    parser.add_argument(
        "--output",
        "--out",
        dest="output",
        type=Path,
        default=Path("field_validation_error.csv"),
    )
    args = parser.parse_args(argv)

    context = load_case(args.config).solver_context
    points = read_csv(args.points)
    _validate_points(points, context.spatial_dim)
    reference = as_long_reference(read_csv(args.reference)) if args.reference else None
    if reference is None:
        quantities = list(
            args.quantities or sorted(context.field_provider.field.quantities.keys())
        )
        rows = _sample_without_reference(context, points, quantities)
        output = pd.DataFrame(rows)
    else:
        rows = _sample_reference_shape(context, points, reference)
        output = merge_with_reference(pd.DataFrame(rows), reference)
    write_csv(output, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
