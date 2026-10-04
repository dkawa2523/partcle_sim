"""Diagnostics derived from an immutable compiled field backend."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .compiled_backend_types import (
    CompiledRuntimeBackend,
    RegularRectilinearCompiledBackend,
)
from .drag_models import drag_model_stage_gas_requirements


def _positive_grid_stats(
    values: np.ndarray,
    valid_mask: np.ndarray | None,
) -> Mapping[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    if valid_mask is None:
        grid = arr.reshape(-1)
        finite = np.isfinite(grid) & (grid > 0.0)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        grid = arr[0] if arr.ndim > mask.ndim else arr
        finite = np.isfinite(grid) & (grid > 0.0)
        if grid.shape == mask.shape:
            finite = mask & finite
    selected = grid[finite]
    if selected.size == 0:
        return {"finite_positive_count": 0}
    return {
        "finite_positive_count": int(selected.size),
        "min": float(np.min(selected)),
        "p50": float(np.percentile(selected, 50.0)),
        "p90": float(np.percentile(selected, 90.0)),
        "max": float(np.max(selected)),
        "mean": float(np.mean(selected)),
    }


def compiled_gas_property_report(
    compiled: CompiledRuntimeBackend,
    *,
    fallback_density_kgm3: float,
    fallback_mu_pas: float,
    fallback_temperature_K: float,
    drag_model_name: str = "",
) -> Mapping[str, object]:
    drag_model = str(drag_model_name).strip().lower()
    drag_requirements = (
        drag_model_stage_gas_requirements(drag_model) if drag_model else ()
    )
    report: dict[str, object] = {
        "field_backend_kind": compiled.backend_kind,
        "drag_model": str(drag_model_name),
        "density_source": compiled.gas_density_source,
        "dynamic_viscosity_source": compiled.gas_mu_source,
        "temperature_source": compiled.gas_temperature_source,
        "fallback_density_kgm3": float(fallback_density_kgm3),
        "fallback_dynamic_viscosity_Pas": float(fallback_mu_pas),
        "fallback_temperature_K": float(fallback_temperature_K),
        "pressure_source": "diagnostic_only_not_used_by_drag",
        "uses_field_density": int(compiled.gas_density_source.startswith("field:")),
        "uses_field_dynamic_viscosity": int(
            compiled.gas_mu_source.startswith("field:")
        ),
        "uses_field_temperature": int(
            compiled.gas_temperature_source.startswith("field:")
        ),
        "density_used_by_drag_model": int("density_kgm3" in drag_requirements),
        "dynamic_viscosity_used_by_drag_model": int(
            "dynamic_viscosity_Pas" in drag_requirements
        ),
        "temperature_used_by_drag_model": int("temperature_K" in drag_requirements),
    }
    if isinstance(compiled, RegularRectilinearCompiledBackend):
        mask = np.asarray(compiled.core_valid_mask, dtype=bool)
        report["density_field_stats"] = dict(
            _positive_grid_stats(compiled.gas_density, mask)
        )
        report["dynamic_viscosity_field_stats"] = dict(
            _positive_grid_stats(compiled.gas_mu, mask)
        )
        report["temperature_field_stats"] = dict(
            _positive_grid_stats(compiled.gas_temperature, mask)
        )
    else:
        report["density_field_stats"] = dict(
            _positive_grid_stats(compiled.gas_density, None)
        )
        report["dynamic_viscosity_field_stats"] = dict(
            _positive_grid_stats(compiled.gas_mu, None)
        )
        report["temperature_field_stats"] = dict(
            _positive_grid_stats(compiled.gas_temperature, None)
        )
        report["triangle_gradient_sources"] = dict(compiled.triangle_gradient_sources)
    return report


__all__ = ("compiled_gas_property_report",)
