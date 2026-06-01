from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.field_backend import sample_field_quantity_with_status
from ..io.runtime_builder import build_prepared_runtime_from_yaml
from ._common import merge_with_reference, read_csv, row_point_id, row_position, row_time, write_csv


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


def _resolve_quantity(field_quantities: set[str], field_name: str, component: str) -> str:
    field = str(field_name).strip()
    comp = str(component).strip()
    candidates = [comp, f"{field}_{comp}", f"{field}{comp}"]
    candidates.extend(_FIELD_COMPONENT_CANDIDATES.get((field, comp), ()))
    candidates.extend(_FIELD_COMPONENT_CANDIDATES.get((field.upper(), comp), ()))
    for candidate in candidates:
        if candidate in field_quantities:
            return candidate
    return comp


def _sample_without_reference(prepared, points, quantities: list[str]) -> list[dict[str, object]]:
    runtime = prepared.runtime
    if runtime.field_provider is None:
        raise ValueError("runtime has no field_provider")
    rows = []
    for fallback_idx, row in points.iterrows():
        pos = row_position(row, runtime.spatial_dim)
        t_eval = row_time(row)
        point_id = row_point_id(row, int(fallback_idx))
        for quantity in quantities:
            sample = sample_field_quantity_with_status(runtime.field_provider, quantity, pos, t_eval)
            rows.append(
                {
                    "point_id": point_id,
                    "time": float(t_eval),
                    "x": float(pos[0]),
                    "y": float(pos[1]) if int(runtime.spatial_dim) >= 2 else np.nan,
                    "z": float(pos[2]) if int(runtime.spatial_dim) == 3 else np.nan,
                    "field": "",
                    "component": str(quantity),
                    "quantity": str(quantity),
                    "python_value": float(sample.value),
                    "provider_kind": sample.provider_kind,
                    "provider_status": int(sample.status),
                    "provider_reason": sample.reason,
                    "cell_id": int(sample.cell_id),
                    "valid": int(bool(sample.valid)),
                }
            )
    return rows


def _sample_reference_shape(prepared, points, reference) -> list[dict[str, object]]:
    runtime = prepared.runtime
    if runtime.field_provider is None:
        raise ValueError("runtime has no field_provider")
    quantities = set(runtime.field_provider.field.quantities.keys())
    point_by_id = {row_point_id(row, int(idx)): row for idx, row in points.iterrows()}
    rows = []
    for fallback_idx, ref_row in reference.iterrows():
        point_id = row_point_id(ref_row, int(fallback_idx))
        point_row = point_by_id.get(point_id, ref_row)
        field_name = str(ref_row.get("field", ""))
        component = str(ref_row.get("component", ref_row.get("quantity", ""))).strip()
        quantity = str(ref_row.get("quantity", "")).strip() or _resolve_quantity(quantities, field_name, component)
        pos = row_position(point_row, runtime.spatial_dim)
        t_eval = row_time(point_row)
        sample = sample_field_quantity_with_status(runtime.field_provider, quantity, pos, t_eval)
        rows.append(
            {
                "point_id": point_id,
                "time": float(t_eval),
                "x": float(pos[0]),
                "y": float(pos[1]) if int(runtime.spatial_dim) >= 2 else np.nan,
                "z": float(pos[2]) if int(runtime.spatial_dim) == 3 else np.nan,
                "field": field_name,
                "component": component or quantity,
                "quantity": quantity,
                "python_value": float(sample.value),
                "provider_kind": sample.provider_kind,
                "provider_status": int(sample.status),
                "provider_reason": sample.reason,
                "cell_id": int(sample.cell_id),
                "valid": int(bool(sample.valid)),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare sampled Python field values against COMSOL CSV samples.")
    parser.add_argument("--config", required=True, type=Path, help="particle_tracer_unified run YAML")
    parser.add_argument("--points", required=True, type=Path, help="CSV with point_id,x,y[,z],time columns")
    parser.add_argument("--comsol", "--reference", dest="reference", type=Path, default=None)
    parser.add_argument("--quantities", nargs="*", default=None, help="Field quantity names to sample when no reference is provided")
    parser.add_argument("--output", "--out", dest="output", type=Path, default=Path("field_validation_error.csv"))
    args = parser.parse_args()

    prepared = build_prepared_runtime_from_yaml(args.config)
    points = read_csv(args.points)
    reference = read_csv(args.reference) if args.reference else None
    if reference is None:
        runtime = prepared.runtime
        if runtime.field_provider is None:
            raise ValueError("runtime has no field_provider")
        quantities = list(args.quantities or sorted(runtime.field_provider.field.quantities.keys()))
        rows = _sample_without_reference(prepared, points, quantities)
        output = pd.DataFrame(rows)
    else:
        rows = _sample_reference_shape(prepared, points, reference)
        output = merge_with_reference(pd.DataFrame(rows), reference)
    write_csv(output, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
