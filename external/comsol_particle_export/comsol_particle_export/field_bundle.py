from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _as_frame(table: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table.copy()
    return pd.read_csv(Path(table))


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"missing required column: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _axis(frame: pd.DataFrame, name: str, scale: float) -> np.ndarray:
    values = _numeric(frame, name)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"axis {name} contains non-finite values")
    axis = np.asarray(sorted(np.unique(values)), dtype=np.float64)
    if axis.size < 2:
        raise ValueError(f"axis {name} must contain at least two unique points")
    if np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"axis {name} must be strictly increasing")
    return axis * float(scale)


def _context_columns(frame: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for aliases in (("time_s", "time", "t"), ("solnum", "solution_number", "solution_index")):
        lower = {str(c).lower(): str(c) for c in frame.columns}
        for alias in aliases:
            if alias == "t":
                col = "t" if "t" in frame.columns else None
            else:
                col = lower.get(alias.lower())
            if col is not None and col not in out:
                out.append(col)
                break
    return out


def _context_key_frame(frame: pd.DataFrame, context_columns: list[str]) -> pd.DataFrame:
    if context_columns:
        return frame[context_columns].drop_duplicates().reset_index(drop=True)
    return pd.DataFrame({"_steady_context": [0]})


def _pivot_quantity(
    frame: pd.DataFrame,
    *,
    axis_names: Sequence[str],
    raw_axes: Sequence[np.ndarray],
    quantity: str,
) -> np.ndarray:
    index_cols = list(axis_names)
    work = frame[index_cols].copy()
    work[quantity] = _numeric(frame, quantity)
    if len(axis_names) == 1:
        out = work.set_index(axis_names[0]).reindex(index=raw_axes[0])[quantity].to_numpy(dtype=np.float64)
        return out
    if len(axis_names) == 2:
        pivot = work.pivot(index=axis_names[0], columns=axis_names[1], values=quantity)
        return pivot.reindex(index=raw_axes[0], columns=raw_axes[1]).to_numpy(dtype=np.float64)
    if len(axis_names) == 3:
        shape = tuple(len(axis) for axis in raw_axes)
        out = np.full(shape, np.nan, dtype=np.float64)
        idx = {name: {float(v): i for i, v in enumerate(axis)} for name, axis in zip(axis_names, raw_axes)}
        values = work[quantity].to_numpy(dtype=np.float64)
        for row_index, row in enumerate(work.itertuples(index=False)):
            coords = [float(getattr(row, name)) for name in axis_names]
            out[tuple(idx[name][coord] for name, coord in zip(axis_names, coords))] = values[row_index]
        return out
    raise ValueError("only 1D, 2D, and 3D field bundles are supported")


def _finite_stats(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    sample = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return {"finite_count": 0}
    return {
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "variation": float(np.max(finite) - np.min(finite)),
    }


def build_field_bundle_from_samples(
    table: str | Path | pd.DataFrame,
    *,
    axis_names: Sequence[str],
    quantities: Sequence[str],
    coordinate_scale_m_per_model_unit: float = 1.0,
    coordinate_model_unit: str = "m",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Build a solver-compatible precomputed NPZ payload from COMSOL samples.

    The input table must contain a complete tensor grid for each optional
    time/solnum context. Quantity arrays are stored as `(nt, *spatial_shape)`.
    The solver currently consumes a time-independent valid mask, so this builder
    uses the intersection of valid nodes across all exported contexts and records
    context-level counts in metadata.
    """

    frame = _as_frame(table)
    axes = [str(name) for name in axis_names]
    fields = [str(name) for name in quantities]
    missing = [name for name in axes + fields if name not in frame.columns]
    if missing:
        raise ValueError(f"field sample table missing required column(s): {', '.join(missing)}")
    scale = float(coordinate_scale_m_per_model_unit)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("coordinate_scale_m_per_model_unit must be a positive finite value")

    raw_axes = [np.asarray(sorted(np.unique(_numeric(frame, name))), dtype=np.float64) for name in axes]
    scaled_axes = [_axis(frame, name, scale) for name in axes]
    expected_per_context = int(np.prod([axis.size for axis in raw_axes]))
    context_columns = _context_columns(frame)
    contexts = _context_key_frame(frame, context_columns)

    quantity_grids: dict[str, list[np.ndarray]] = {name: [] for name in fields}
    valid_grids: list[np.ndarray] = []
    context_records: list[dict[str, Any]] = []
    if context_columns:
        grouped = frame.groupby(context_columns, dropna=False, sort=False)
        context_iter = list(grouped)
    else:
        context_iter = [(("_steady_context",), frame)]

    for key, sub in context_iter:
        if int(len(sub)) != expected_per_context:
            raise ValueError(
                "field sample table must contain a complete tensor grid for every context; "
                f"got {len(sub)} rows, expected {expected_per_context}"
            )
        if sub.duplicated(axes).any():
            raise ValueError("field sample table contains duplicate coordinates within a context")
        if "valid_mask" in sub.columns:
            valid = _pivot_quantity(sub, axis_names=axes, raw_axes=raw_axes, quantity="valid_mask") > 0.5
        else:
            valid = np.ones(tuple(axis.size for axis in raw_axes), dtype=bool)
        for name in fields:
            grid = _pivot_quantity(sub, axis_names=axes, raw_axes=raw_axes, quantity=name)
            if np.any(~np.isfinite(grid[valid])):
                raise ValueError(f"quantity {name} is non-finite on valid support")
            quantity_grids[name].append(grid)
        valid_grids.append(np.asarray(valid, dtype=bool))
        if context_columns:
            if not isinstance(key, tuple):
                key = (key,)
            context_records.append({name: value for name, value in zip(context_columns, key)})
        else:
            context_records.append({})

    valid_mask = np.logical_and.reduce(valid_grids)
    if not np.any(valid_mask):
        raise ValueError("valid_mask intersection across contexts contains no valid nodes")

    time_values = np.asarray([0.0], dtype=np.float64)
    time_col = next((col for col in context_columns if col.lower() in {"time_s", "time", "t"}), None)
    if time_col is not None:
        time_values = pd.to_numeric(contexts[time_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    elif len(context_records) > 1:
        time_values = np.arange(len(context_records), dtype=np.float64)

    payload: dict[str, np.ndarray] = {
        f"axis_{i}": axis.astype(np.float64) for i, axis in enumerate(scaled_axes)
    }
    payload["times"] = time_values.astype(np.float64)
    payload["valid_mask"] = valid_mask.astype(bool)

    quantity_summary: dict[str, Any] = {}
    for name, grids in quantity_grids.items():
        arr = np.stack(grids, axis=0).astype(np.float64)
        arr[:, ~valid_mask] = np.nan
        payload[name] = arr
        quantity_summary[name] = _finite_stats(arr[0], valid_mask)

    meta = dict(metadata or {})
    meta.update(
        {
            "source_kind": "external_comsol_particle_export_field_bundle",
            "axis_names": axes,
            "quantities": fields,
            "coordinate_model_unit": str(coordinate_model_unit),
            "coordinate_scale_m_per_model_unit": scale,
            "grid_shape": [int(axis.size) for axis in scaled_axes],
            "time_count": int(time_values.size),
            "sample_context_columns": context_columns,
            "sample_contexts": context_records,
            "valid_node_count": int(np.count_nonzero(valid_mask)),
            "quantity_summary": quantity_summary,
        }
    )
    payload["metadata_json"] = np.asarray(json.dumps(meta))
    return payload


def _validate_mesh_arrays(mesh_vertices: np.ndarray, mesh_triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh_vertices, dtype=np.float64)
    triangles = np.asarray(mesh_triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 2:
        raise ValueError("mesh_vertices must have shape (n, 2) for triangle mesh field bundles")
    if vertices.shape[0] < 3:
        raise ValueError("mesh_vertices must contain at least three vertices")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("mesh_triangles must have shape (m, 3)")
    if triangles.shape[0] < 1:
        raise ValueError("mesh_triangles must contain at least one triangle")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("mesh_vertices must contain only finite values")
    if int(np.min(triangles)) < 0 or int(np.max(triangles)) >= int(vertices.shape[0]):
        raise ValueError("mesh_triangles contains vertex indices outside mesh_vertices")
    return vertices, triangles.astype(np.int32)


def _vertex_ids(frame: pd.DataFrame) -> np.ndarray:
    col = "vertex_id"
    if col not in frame.columns:
        raise ValueError("mesh field sample table missing required column: vertex_id")
    values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
    if np.any(~np.isfinite(values)):
        raise ValueError("vertex_id contains non-finite values")
    rounded = np.rint(values).astype(np.int64)
    if np.any(np.abs(values - rounded.astype(np.float64)) > 0.0):
        raise ValueError("vertex_id must contain integer values")
    return rounded


def build_triangle_mesh_field_bundle_from_samples(
    table: str | Path | pd.DataFrame,
    *,
    mesh_vertices: np.ndarray,
    mesh_triangles: np.ndarray,
    axis_names: Sequence[str],
    quantities: Sequence[str],
    coordinate_scale_m_per_model_unit: float = 1.0,
    coordinate_model_unit: str = "m",
    metadata: Mapping[str, Any] | None = None,
    coordinate_tolerance_m: float = 1.0e-9,
) -> dict[str, np.ndarray]:
    """Build a solver-compatible triangle-mesh field NPZ from COMSOL samples.

    The sample table must contain one row per mesh vertex for every optional
    time context. Rows are reordered by the zero-based ``vertex_id`` column, so
    the emitted nodal arrays align exactly with ``mesh_vertices`` and
    ``mesh_triangles``.
    """

    frame = _as_frame(table)
    axes = [str(name) for name in axis_names]
    if len(axes) != 2:
        raise ValueError("triangle mesh field bundles currently require exactly two axis names")
    fields = [str(name) for name in quantities]
    missing = [name for name in ["vertex_id", *axes, *fields] if name not in frame.columns]
    if missing:
        raise ValueError(f"mesh field sample table missing required column(s): {', '.join(missing)}")

    scale = float(coordinate_scale_m_per_model_unit)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("coordinate_scale_m_per_model_unit must be a positive finite value")
    tolerance = float(coordinate_tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("coordinate_tolerance_m must be finite and non-negative")

    vertices, triangles = _validate_mesh_arrays(mesh_vertices, mesh_triangles)
    vertex_count = int(vertices.shape[0])
    context_columns = _context_columns(frame)
    if context_columns:
        context_iter = list(frame.groupby(context_columns, dropna=False, sort=False))
    else:
        context_iter = [(("_steady_context",), frame)]

    time_col = next((col for col in context_columns if col.lower() in {"time_s", "time", "t"}), None)
    if len(context_iter) > 1 and time_col is None:
        raise ValueError("mesh field samples with multiple contexts require a time_s/time/t column")

    quantity_series: dict[str, list[np.ndarray]] = {name: [] for name in fields}
    context_records: list[dict[str, Any]] = []
    time_values: list[float] = []
    invalid_counts: list[int] = []
    seen_contexts = 0
    expected_vertex_ids = np.arange(vertex_count, dtype=np.int64)

    for key, sub_raw in context_iter:
        seen_contexts += 1
        sub = sub_raw.copy()
        ids = _vertex_ids(sub)
        if int(len(sub)) != vertex_count:
            raise ValueError(
                "mesh field sample table must contain one row per mesh vertex for every context; "
                f"got {len(sub)} rows, expected {vertex_count}"
            )
        if len(np.unique(ids)) != vertex_count:
            raise ValueError("mesh field sample table contains duplicate vertex_id values within a context")
        if int(np.min(ids)) != 0 or int(np.max(ids)) != vertex_count - 1:
            raise ValueError("mesh field sample table vertex_id values must cover 0..n_vertices-1")

        order = np.argsort(ids)
        ordered = sub.iloc[order].reset_index(drop=True)
        ordered_ids = _vertex_ids(ordered)
        if not np.array_equal(ordered_ids, expected_vertex_ids):
            raise ValueError("mesh field sample table vertex_id values must cover 0..n_vertices-1")

        coords = np.column_stack([_numeric(ordered, name) for name in axes]) * scale
        if np.any(~np.isfinite(coords)):
            raise ValueError("mesh field sample coordinates contain non-finite values")
        if not np.allclose(coords, vertices, rtol=0.0, atol=tolerance):
            max_delta = float(np.max(np.linalg.norm(coords - vertices, axis=1)))
            raise ValueError(
                "mesh field sample coordinates do not match mesh_vertices after scaling; "
                f"max_delta_m={max_delta:.6g}, tolerance_m={tolerance:.6g}"
            )

        if "valid_mask" in ordered.columns:
            valid = pd.to_numeric(ordered["valid_mask"], errors="coerce").to_numpy(dtype=np.float64) > 0.5
            invalid_count = int(np.count_nonzero(~valid))
        else:
            valid = np.ones(vertex_count, dtype=bool)
            invalid_count = 0
        invalid_counts.append(invalid_count)
        if invalid_count:
            raise ValueError(f"mesh field samples contain {invalid_count} invalid mesh vertices in context {seen_contexts}")

        for name in fields:
            values = _numeric(ordered, name)
            if np.any(~np.isfinite(values)):
                raise ValueError(f"quantity {name} is non-finite on mesh vertices")
            quantity_series[name].append(values.astype(np.float64))

        if context_columns:
            if not isinstance(key, tuple):
                key = (key,)
            record = {name: value for name, value in zip(context_columns, key)}
        else:
            record = {}
        context_records.append(record)
        if time_col is not None:
            t = pd.to_numeric(pd.Series([record.get(time_col)]), errors="coerce").to_numpy(dtype=np.float64)[0]
            if not np.isfinite(t):
                raise ValueError(f"non-finite mesh field time value in context {seen_contexts}")
            time_values.append(float(t))
        else:
            time_values.append(0.0)

    times = np.asarray(time_values, dtype=np.float64)
    if times.size != seen_contexts:
        raise ValueError("internal error while collecting mesh field time contexts")
    if times.size > 1 and np.any(np.diff(times) < 0.0):
        raise ValueError("mesh field time values must be non-decreasing")

    payload: dict[str, np.ndarray] = {
        "mesh_vertices": vertices.astype(np.float64),
        "mesh_triangles": triangles.astype(np.int32),
        "times": times.astype(np.float64),
    }
    quantity_summary: dict[str, Any] = {}
    for name, values_by_context in quantity_series.items():
        arr = np.stack(values_by_context, axis=0).astype(np.float64)
        payload[name] = arr
        quantity_summary[name] = _finite_stats(arr[0], np.ones(vertex_count, dtype=bool))

    meta = dict(metadata or {})
    meta.update(
        {
            "source_kind": "external_comsol_particle_export_triangle_mesh_field_bundle",
            "field_backend_kind": "triangle_mesh_2d",
            "axis_names": axes,
            "quantities": fields,
            "coordinate_model_unit": str(coordinate_model_unit),
            "coordinate_scale_m_per_model_unit": scale,
            "mesh_vertex_count": vertex_count,
            "mesh_triangle_count": int(triangles.shape[0]),
            "time_count": int(times.size),
            "sample_context_columns": context_columns,
            "sample_contexts": context_records,
            "invalid_vertex_counts": invalid_counts,
            "quantity_summary": quantity_summary,
        }
    )
    payload["metadata_json"] = np.asarray(json.dumps(meta))
    return payload


def write_field_bundle(bundle: Mapping[str, np.ndarray], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{key: np.asarray(value) for key, value in bundle.items()})
