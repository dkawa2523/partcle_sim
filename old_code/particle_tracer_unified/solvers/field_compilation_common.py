from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from particle_tracer_unified.core.field_sampling import as_time_grid


@dataclass(frozen=True, slots=True)
class GasDefaults:
    density_kgm3: float
    dynamic_viscosity_Pas: float
    temperature_K: float
    density_source: str
    viscosity_source: str
    temperature_source: str


def require_positive_declared_gas_grid(
    values: np.ndarray,
    support_mask: np.ndarray,
    *,
    semantic_name: str,
    quantity_name: str,
) -> None:
    """Reject invalid declared gas values once, before entering solver loops."""

    grid = np.asarray(values, dtype=np.float64)
    support = np.asarray(support_mask, dtype=bool)
    if grid.ndim != support.ndim + 1 or grid.shape[1:] != support.shape:
        raise ValueError(
            f"Field gas property {semantic_name} ({quantity_name}) shape must "
            "match solver support"
        )
    supported_values = grid[:, support]
    invalid = ~np.isfinite(supported_values) | (supported_values <= 0.0)
    if np.any(invalid):
        raise ValueError(
            f"Field gas property {semantic_name} ({quantity_name}) must be "
            "finite and > 0 inside solver support"
        )


def _positive_source(value: float) -> str:
    return "context:gas" if np.isfinite(value) and value > 0.0 else "unavailable"


def gas_defaults(runtime: Any) -> GasDefaults:
    gas = getattr(runtime, "gas", None)
    density = float(getattr(gas, "density_kgm3", np.nan))
    viscosity = float(getattr(gas, "dynamic_viscosity_Pas", np.nan))
    temperature = float(getattr(gas, "temperature", np.nan))
    return GasDefaults(
        density_kgm3=density,
        dynamic_viscosity_Pas=viscosity,
        temperature_K=temperature,
        density_source=_positive_source(density),
        viscosity_source=_positive_source(viscosity),
        temperature_source=_positive_source(temperature),
    )


def backend_time_grid(
    data: np.ndarray, spatial_dim: int, times: np.ndarray
) -> np.ndarray:
    grid = as_time_grid(data, int(spatial_dim))
    time_count = int(max(1, np.asarray(times, dtype=np.float64).size))
    if grid.shape[0] == 1 and time_count > 1:
        return np.repeat(grid, time_count, axis=0)
    return grid


def zero_like_grid(reference: np.ndarray) -> np.ndarray:
    return np.zeros_like(np.asarray(reference, dtype=np.float64), dtype=np.float64)


def gradient_time_grid(
    data: np.ndarray,
    axes: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != len(axes) + 1:
        raise ValueError(
            "gradient source must be a time grid with shape (nt, ...spatial axes)"
        )
    spatial_axes = tuple(np.asarray(axis, dtype=np.float64) for axis in axes)
    if any(axis.size < 2 for axis in spatial_axes):
        return tuple(zero_like_grid(arr) for _ in spatial_axes)
    edge_order = 2 if all(axis.size >= 3 for axis in spatial_axes) else 1
    grads = np.gradient(
        arr,
        *spatial_axes,
        axis=tuple(range(1, arr.ndim)),
        edge_order=edge_order,
    )
    return tuple(np.asarray(grad, dtype=np.float64) for grad in grads)


def time_derivative_time_grid(data: np.ndarray, times: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    time_grid = np.asarray(times, dtype=np.float64)
    if arr.ndim < 1 or arr.shape[0] <= 1 or time_grid.size <= 1:
        return zero_like_grid(arr)
    edge_order = 2 if arr.shape[0] >= 3 else 1
    return np.asarray(
        np.gradient(arr, time_grid, axis=0, edge_order=edge_order),
        dtype=np.float64,
    )


def vertex_time_grid(data: np.ndarray, times: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    time_count = int(max(1, np.asarray(times, dtype=np.float64).size))
    if arr.ndim == 1:
        arr = arr.reshape(1, arr.shape[0])
    if arr.shape[0] == 1 and time_count > 1:
        return np.repeat(arr, time_count, axis=0)
    return arr


def curl_from_velocity_grids(
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray | None,
    axes: tuple[np.ndarray, ...],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if len(axes) == 2:
        dux_dx, dux_dy = gradient_time_grid(ux, axes)
        duy_dx, _duy_dy = gradient_time_grid(uy, axes)
        return None, None, np.asarray(duy_dx - dux_dy, dtype=np.float64)
    if uz is None:
        return None, None, None
    dux_dx, dux_dy, dux_dz = gradient_time_grid(ux, axes)
    duy_dx, duy_dy, duy_dz = gradient_time_grid(uy, axes)
    duz_dx, duz_dy, duz_dz = gradient_time_grid(uz, axes)
    _ = (dux_dx, duy_dy, duz_dz)
    return (
        np.asarray(duz_dy - duy_dz, dtype=np.float64),
        np.asarray(dux_dz - duz_dx, dtype=np.float64),
        np.asarray(duy_dx - dux_dy, dtype=np.float64),
    )


def common_quantity_times(field: Any, quantity_names: tuple[str, ...]) -> np.ndarray:
    times = np.asarray([0.0], dtype=np.float64)
    first_name = ""
    for name in quantity_names:
        series = field.quantities.get(str(name))
        if series is None:
            continue
        current = np.asarray(series.times, dtype=np.float64)
        if current.size == 0:
            current = np.asarray([0.0], dtype=np.float64)
        if not first_name:
            first_name = str(name)
            times = current
            continue
        if current.shape != times.shape or not np.allclose(
            current, times, rtol=0.0, atol=0.0
        ):
            raise ValueError(
                "Field quantities used by the solver must share one time axis; "
                f"{first_name} and {name} differ"
            )
    return times


def merge_optional_quantity_times(
    field: Any,
    base_times: np.ndarray,
    quantity_names: tuple[str, ...],
) -> np.ndarray:
    times = np.asarray(base_times, dtype=np.float64)
    if times.size == 0:
        times = np.asarray([0.0], dtype=np.float64)
    first_transient_name = ""
    for name in quantity_names:
        series = field.quantities.get(str(name))
        if series is None:
            continue
        current = np.asarray(series.times, dtype=np.float64)
        if current.size == 0:
            current = np.asarray([0.0], dtype=np.float64)
        if current.size <= 1:
            continue
        if times.size <= 1:
            times = current
            first_transient_name = str(name)
            continue
        if current.shape != times.shape or not np.allclose(
            current, times, rtol=0.0, atol=0.0
        ):
            reference = first_transient_name or "primary solver quantities"
            message = (
                "Field quantities used by the solver must share one transient time axis"
            )
            raise ValueError(f"{message}; {reference} and {name} differ")
    return times


def gas_property_quantity_names(field: Any) -> Mapping[str, str]:
    """Resolve canonical gas properties from the field quantity inventory."""

    selected: dict[str, str] = {}
    for candidates, target in (
        (("rho_g", "gas_density", "density_kgm3", "rho"), "gas_density"),
        (("mu", "dynamic_viscosity", "dynamic_viscosity_Pas"), "gas_mu"),
        (("T", "temperature", "temperature_K", "gas_temperature"), "gas_temperature"),
    ):
        for name in candidates:
            if name in field.quantities:
                selected[target] = str(name)
                break
    return selected
