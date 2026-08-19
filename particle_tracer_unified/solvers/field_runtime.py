from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from time import perf_counter
from typing import cast

import numpy as np

from particle_tracer_unified.domain import FieldRequest, StageFields

from .compiled_backend_types import CompiledRuntimeBackend
from .runtime_plan import StageFieldPlan
from .sampling_backend import (
    DYNAMIC_VISCOSITY,
    ELECTRIC_FIELD,
    FLOW_VELOCITY,
    GAS_DENSITY,
    TEMPERATURE,
    VALID_MASK_STATUS,
    CompiledSamplingBackend,
)


@dataclass(frozen=True)
class FieldSamplingMetrics:
    """Debug-only measurements kept separate from sampled field values."""

    elapsed_s: float
    call_count: int
    point_count: int


def _explicit_or_planned(explicit: bool | None, planned: bool) -> bool:
    return bool(planned) if explicit is None else bool(explicit)


def _requested_gas_fields(
    field_plan: StageFieldPlan | None,
    *,
    all_gas: bool | None,
    density: bool | None,
    viscosity: bool | None,
    temperature: bool | None,
) -> tuple[bool, bool, bool]:
    if any(value is not None for value in (density, viscosity, temperature)):
        return bool(density), bool(viscosity), bool(temperature)
    if all_gas is not None:
        requested = bool(all_gas)
        return requested, requested, requested
    if field_plan is None:
        return False, False, False
    return (
        bool(field_plan.need_gas_density),
        bool(field_plan.need_gas_mu),
        bool(field_plan.need_gas_temperature),
    )


def _requested_quantities(
    field_plan: StageFieldPlan | None,
    *,
    need_flow: bool | None,
    need_electric: bool | None,
    need_gas_properties: bool | None,
    need_gas_density: bool | None,
    need_gas_mu: bool | None,
    need_gas_temperature: bool | None,
    need_valid_mask: bool | None,
) -> tuple[str, ...]:
    planned_flow = bool(field_plan is not None and field_plan.need_flow)
    planned_electric = bool(field_plan is not None and field_plan.need_electric)
    planned_mask = bool(field_plan is not None and field_plan.need_valid_mask)
    gas_density, gas_mu, gas_temperature = _requested_gas_fields(
        field_plan,
        all_gas=need_gas_properties,
        density=need_gas_density,
        viscosity=need_gas_mu,
        temperature=need_gas_temperature,
    )
    requested = (
        (FLOW_VELOCITY, _explicit_or_planned(need_flow, planned_flow)),
        (ELECTRIC_FIELD, _explicit_or_planned(need_electric, planned_electric)),
        (GAS_DENSITY, gas_density),
        (DYNAMIC_VISCOSITY, gas_mu),
        (TEMPERATURE, gas_temperature),
        (VALID_MASK_STATUS, _explicit_or_planned(need_valid_mask, planned_mask)),
    )
    return tuple(name for name, enabled in requested if enabled)


def sample_fields_for_stage(
    compiled: CompiledRuntimeBackend,
    field_plan: StageFieldPlan | None,
    points: np.ndarray,
    time_s: float,
    *,
    spatial_dim: int,
    need_flow: bool | None = None,
    need_electric: bool | None = None,
    need_gas_properties: bool | None = None,
    need_gas_density: bool | None = None,
    need_gas_mu: bool | None = None,
    need_gas_temperature: bool | None = None,
    need_valid_mask: bool | None = None,
    fallback_density_kgm3: float = float("nan"),
    fallback_mu_pas: float = float("nan"),
    fallback_temperature_K: float = float("nan"),
) -> StageFields:
    """Adapt solver-plan flags to one semantic batch request."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must have shape (n, spatial_dim)")
    dim = int(spatial_dim)
    if pts.shape[1] != dim:
        raise ValueError("points must have shape (n, spatial_dim)")

    quantities = _requested_quantities(
        field_plan,
        need_flow=need_flow,
        need_electric=need_electric,
        need_gas_properties=need_gas_properties,
        need_gas_density=need_gas_density,
        need_gas_mu=need_gas_mu,
        need_gas_temperature=need_gas_temperature,
        need_valid_mask=need_valid_mask,
    )

    backend = CompiledSamplingBackend(
        compiled=compiled,
        spatial_dim=dim,
        fallback_density_kgm3=float(fallback_density_kgm3),
        fallback_dynamic_viscosity_Pas=float(fallback_mu_pas),
        fallback_temperature_K=float(fallback_temperature_K),
        strict=False,
    )
    request = FieldRequest(quantities or (VALID_MASK_STATUS,))
    stage = backend.sample(pts, float(time_s), request)
    if not quantities:
        stage = StageFields(
            points_m=stage.points_m,
            time_s=stage.time_s,
            values={},
            supported=stage.supported,
            metadata=stage.metadata,
        )
    return stage


def sample_scalar_fields_for_stage(
    compiled: CompiledRuntimeBackend,
    field_plan: StageFieldPlan | None,
    position: np.ndarray,
    time_s: float,
    *,
    spatial_dim: int,
    need_flow: bool | None = None,
    need_electric: bool | None = None,
    need_gas_properties: bool | None = None,
    need_gas_density: bool | None = None,
    need_gas_mu: bool | None = None,
    need_gas_temperature: bool | None = None,
    need_valid_mask: bool | None = None,
    fallback_density_kgm3: float = float("nan"),
    fallback_mu_pas: float = float("nan"),
    fallback_temperature_K: float = float("nan"),
) -> StageFields:
    """Sample one point by delegating to the canonical batch implementation."""

    pos = np.asarray(position, dtype=np.float64)
    if pos.ndim != 1 or pos.shape[0] != int(spatial_dim):
        raise ValueError("position must have shape (spatial_dim,)")
    return sample_fields_for_stage(
        compiled,
        field_plan,
        pos.reshape(1, int(spatial_dim)),
        float(time_s),
        spatial_dim=int(spatial_dim),
        need_flow=need_flow,
        need_electric=need_electric,
        need_gas_properties=need_gas_properties,
        need_gas_density=need_gas_density,
        need_gas_mu=need_gas_mu,
        need_gas_temperature=need_gas_temperature,
        need_valid_mask=need_valid_mask,
        fallback_density_kgm3=float(fallback_density_kgm3),
        fallback_mu_pas=float(fallback_mu_pas),
        fallback_temperature_K=float(fallback_temperature_K),
    )


def measure_sample_fields_for_stage(
    *args, **kwargs
) -> tuple[StageFields, FieldSamplingMetrics]:
    """Sample fields and measure the call for explicit debug diagnostics."""

    start = perf_counter()
    samples = sample_fields_for_stage(*args, **kwargs)
    metrics = FieldSamplingMetrics(
        elapsed_s=float(perf_counter() - start),
        call_count=int(samples.metadata.get("sample_call_count", 0)),
        point_count=int(samples.metadata.get("sample_point_count", 0)),
    )
    return samples, metrics


def record_field_sampling_diagnostics(
    diagnostics: MutableMapping[str, object],
    metrics: FieldSamplingMetrics,
) -> None:
    diagnostics["field_sampling_s"] = float(
        cast(float, diagnostics.get("field_sampling_s", 0.0))
    ) + float(metrics.elapsed_s)
    diagnostics["field_sample_point_count"] = int(
        cast(int, diagnostics.get("field_sample_point_count", 0))
    ) + int(metrics.point_count)
    diagnostics["field_sample_call_count"] = int(
        cast(int, diagnostics.get("field_sample_call_count", 0))
    ) + int(metrics.call_count)


__all__ = (
    "FieldSamplingMetrics",
    "measure_sample_fields_for_stage",
    "record_field_sampling_diagnostics",
    "sample_fields_for_stage",
    "sample_scalar_fields_for_stage",
)
