from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from particle_tracer_unified.core.numerical_contracts import (
    float_arrays_equal_ulps,
    float_values_equal_ulps,
)

_RESERVED_ARRAYS = {
    "axis_0",
    "axis_1",
    "times",
    "valid_mask",
    "support_phi",
    "metadata_json",
}


@dataclass(frozen=True)
class _BundleGrid:
    source_x: np.ndarray
    source_y: np.ndarray
    target_x: np.ndarray
    target_y: np.ndarray
    source_shape: tuple[int, int]
    target_shape: tuple[int, int]
    times: np.ndarray
    resample: bool


def _load_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _axis_summary(axis: np.ndarray | None) -> dict[str, Any]:
    if axis is None:
        return {"count": 0, "min": None, "max": None}
    array = np.asarray(axis, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "count": int(array.size),
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
    }


def _axes_match(source: np.ndarray | None, target: np.ndarray) -> bool:
    return source is None or float_arrays_equal_ulps(source, target)


def _same_axis_extent(source: np.ndarray, target: np.ndarray) -> bool:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return bool(
        source.ndim == 1
        and target.ndim == 1
        and source.size > 0
        and target.size > 0
        and float_values_equal_ulps(source[0], target[0])
        and float_values_equal_ulps(source[-1], target[-1])
    )


def _field_axis_alignment_summary(
    payload: Mapping[str, np.ndarray], axes_x: np.ndarray, axes_y: np.ndarray
) -> dict[str, Any]:
    source_x = (
        np.asarray(payload["axis_0"], np.float64) if "axis_0" in payload else None
    )
    source_y = (
        np.asarray(payload["axis_1"], np.float64) if "axis_1" in payload else None
    )
    axis_0_match = _axes_match(source_x, axes_x)
    axis_1_match = _axes_match(source_y, axes_y)
    exact_match = axis_0_match and axis_1_match
    return {
        "source_axes": {
            "axis_0": _axis_summary(source_x),
            "axis_1": _axis_summary(source_y),
        },
        "geometry_axes": {
            "axis_0": _axis_summary(axes_x),
            "axis_1": _axis_summary(axes_y),
        },
        "axis_0_exact_match": bool(axis_0_match),
        "axis_1_exact_match": bool(axis_1_match),
        "exact_match": bool(exact_match),
        "resampled_to_geometry_axes": not exact_match,
    }


def _interpolate_2d(
    data: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    along_x = np.empty((target_x.size, source_y.size), dtype=np.float64)
    for index in range(source_y.size):
        along_x[:, index] = np.interp(target_x, source_x, data[:, index])
    result = np.empty((target_x.size, target_y.size), dtype=np.float64)
    for index in range(target_x.size):
        result[index] = np.interp(target_y, source_y, along_x[index])
    return result


def _resample(
    data: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> np.ndarray:
    data = np.asarray(data)
    if data.ndim == 2:
        return _interpolate_2d(data, source_x, source_y, target_x, target_y)
    if data.ndim == 3:
        return np.stack(
            [
                _interpolate_2d(layer, source_x, source_y, target_x, target_y)
                for layer in data
            ],
            axis=0,
        )
    raise ValueError(f"field bundle quantity must be 2D or 3D, got shape {data.shape}")


def _source_axis(
    payload: Mapping[str, np.ndarray],
    name: str,
) -> np.ndarray | None:
    if name not in payload:
        return None
    return np.asarray(payload[name], dtype=np.float64)


def _validate_resampling_axes(
    source_x: np.ndarray | None,
    source_y: np.ndarray | None,
    target_x: np.ndarray,
    target_y: np.ndarray,
) -> None:
    if source_x is None or source_y is None:
        raise ValueError(
            "field bundle axes are required when resampling to geometry axes"
        )
    if not _same_axis_extent(source_x, target_x):
        raise ValueError(
            "field bundle axis_0 must share geometry axis_0 extent to be resampled"
        )
    if not _same_axis_extent(source_y, target_y):
        raise ValueError(
            "field bundle axis_1 must share geometry axis_1 extent to be resampled"
        )


def _bundle_times(payload: Mapping[str, np.ndarray]) -> np.ndarray:
    times = np.asarray(payload.get("times", [0.0]), dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        raise ValueError(
            "field bundle times must be a non-empty 1D array when provided"
        )
    return times


def _bundle_grid(
    payload: Mapping[str, np.ndarray],
    axes_x: np.ndarray,
    axes_y: np.ndarray,
) -> _BundleGrid:
    source_x = _source_axis(payload, "axis_0")
    source_y = _source_axis(payload, "axis_1")
    resample = not (_axes_match(source_x, axes_x) and _axes_match(source_y, axes_y))
    if resample:
        _validate_resampling_axes(source_x, source_y, axes_x, axes_y)
    resolved_x = axes_x if source_x is None else source_x
    resolved_y = axes_y if source_y is None else source_y
    return _BundleGrid(
        source_x=resolved_x,
        source_y=resolved_y,
        target_x=axes_x,
        target_y=axes_y,
        source_shape=(resolved_x.size, resolved_y.size),
        target_shape=(axes_x.size, axes_y.size),
        times=_bundle_times(payload),
        resample=resample,
    )


def _normalize_valid_mask(
    payload: Mapping[str, np.ndarray],
    grid: _BundleGrid,
) -> np.ndarray:
    valid_mask = np.asarray(
        payload.get("valid_mask", np.ones(grid.source_shape, dtype=bool)),
        dtype=bool,
    )
    if grid.resample:
        return (
            _resample(
                valid_mask.astype(np.float64),
                grid.source_x,
                grid.source_y,
                grid.target_x,
                grid.target_y,
            )
            >= 0.5
        )
    if valid_mask.shape != grid.target_shape:
        raise ValueError(
            "field bundle valid_mask must match geometry grid shape "
            f"{grid.target_shape}"
        )
    return valid_mask


def _normalize_quantity(
    name: str,
    value: np.ndarray,
    grid: _BundleGrid,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if grid.resample:
        return _resample(
            array,
            grid.source_x,
            grid.source_y,
            grid.target_x,
            grid.target_y,
        )
    if name == "support_phi" and array.shape != grid.target_shape:
        raise ValueError(
            "field bundle support_phi must match geometry grid shape "
            f"{grid.target_shape}"
        )
    if array.ndim == 2 and array.shape != grid.target_shape:
        raise ValueError(
            f"field bundle quantity {name} must match geometry grid shape "
            f"{grid.target_shape}"
        )
    if array.ndim == 3 and (
        array.shape[0] != grid.times.size or array.shape[1:] != grid.target_shape
    ):
        raise ValueError(
            f"field bundle quantity {name} must match shape "
            f"{(grid.times.size, *grid.target_shape)}"
        )
    if array.ndim not in {2, 3}:
        raise ValueError(
            f"field bundle quantity {name} must be 2D or 3D, got shape {array.shape}"
        )
    return array


def _normalize_bundle(
    payload: Mapping[str, np.ndarray], axes_x: np.ndarray, axes_y: np.ndarray
) -> dict[str, np.ndarray]:
    if "ux" not in payload or "uy" not in payload:
        raise ValueError("field bundle must include ux and uy")
    grid = _bundle_grid(payload, axes_x, axes_y)
    normalized: dict[str, np.ndarray] = {
        "axis_0": axes_x.astype(np.float64),
        "axis_1": axes_y.astype(np.float64),
        "times": grid.times,
        "valid_mask": _normalize_valid_mask(payload, grid),
    }
    for name, value in payload.items():
        if name in {"axis_0", "axis_1", "times", "valid_mask", "metadata_json"}:
            continue
        normalized[name] = _normalize_quantity(name, value, grid)
    return normalized
