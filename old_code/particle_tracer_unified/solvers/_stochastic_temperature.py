"""Resolve gas or field temperatures for accepted stochastic-motion leaves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._stochastic_config import StochasticMotionConfig
from .compiled_backend_types import CompiledRuntimeBackend
from .field_runtime import measure_sample_fields_for_stage, sample_fields_for_stage
from .sampling_backend import TEMPERATURE


@dataclass(frozen=True, slots=True)
class ParticleLeafPlan:
    particle_index: int
    leaf_end_times_s: np.ndarray
    midpoint_times_s: np.ndarray
    midpoint_positions_m: np.ndarray
    tau_mid_s: np.ndarray
    particle_mass_kg: float


def _temperature_source(
    config: StochasticMotionConfig,
    compiled: CompiledRuntimeBackend,
) -> tuple[str, bool]:
    mode = str(config.temperature_source)
    if mode not in {"field_T_then_gas", "gas"}:
        raise ValueError("Brownian temperature_source must be field_T_then_gas or gas")
    declared = str(compiled.gas_temperature_source)
    return declared, bool(mode == "field_T_then_gas" and declared.startswith("field:"))


def _configured_plan_temperatures(
    plans: list[ParticleLeafPlan],
    gas_temperature_K: float,
) -> list[np.ndarray]:
    temperature = float(gas_temperature_K)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(
            "Brownian motion requires finite positive configured gas temperature "
            "when no temperature field is selected"
        )
    return [
        np.full(plan.leaf_end_times_s.size, temperature, dtype=np.float64)
        for plan in plans
    ]


def _group_temperature_samples(
    plans: list[ParticleLeafPlan],
) -> dict[float, list[tuple[int, int, np.ndarray]]]:
    grouped: dict[float, list[tuple[int, int, np.ndarray]]] = {}
    for plan_row, plan in enumerate(plans):
        for leaf_row, time_s in enumerate(plan.midpoint_times_s):
            grouped.setdefault(float(time_s), []).append(
                (
                    plan_row,
                    leaf_row,
                    np.asarray(plan.midpoint_positions_m[leaf_row], dtype=np.float64),
                )
            )
    return grouped


def _sample_temperature_group(
    *,
    compiled: CompiledRuntimeBackend,
    entries: list[tuple[int, int, np.ndarray]],
    time_s: float,
    spatial_dim: int,
    gas_temperature_K: float,
    collect_diagnostics: bool,
) -> tuple[np.ndarray, float, int, int]:
    points = np.asarray([entry[2] for entry in entries], dtype=np.float64)
    args = (compiled, None, points, float(time_s))
    kwargs = {
        "spatial_dim": int(spatial_dim),
        "need_gas_temperature": True,
        "need_valid_mask": False,
        "fallback_temperature_K": float(gas_temperature_K),
    }
    if collect_diagnostics:
        sampled, metrics = measure_sample_fields_for_stage(*args, **kwargs)
        diagnostics = (
            float(metrics.elapsed_s),
            int(metrics.point_count),
            int(metrics.call_count),
        )
    else:
        sampled = sample_fields_for_stage(*args, **kwargs)
        diagnostics = (0.0, 0, 0)
    values = sampled.values.get(TEMPERATURE)
    if values is None:
        raise ValueError(
            "Brownian temperature field was declared but was not returned by the "
            "sampling backend"
        )
    temperatures = np.asarray(values, dtype=np.float64)
    if temperatures.shape != (len(entries),):
        raise ValueError(
            f"Brownian temperature field must have shape {(len(entries),)}, "
            f"got {temperatures.shape}"
        )
    return temperatures, diagnostics[0], diagnostics[1], diagnostics[2]


def _validate_plan_temperatures(
    plans: list[ParticleLeafPlan],
    output: list[np.ndarray],
    declared_source: str,
) -> None:
    invalid_particles = sorted(
        {
            int(plan.particle_index)
            for plan, values in zip(plans, output, strict=True)
            if np.any(~np.isfinite(values) | (values <= 0.0))
        }
    )
    if invalid_particles:
        raise ValueError(
            f"Brownian declared temperature field {declared_source!r} must be finite "
            f"and positive; invalid particle indices: {invalid_particles}"
        )


def _sample_field_plan_temperatures(
    *,
    compiled: CompiledRuntimeBackend,
    plans: list[ParticleLeafPlan],
    declared_source: str,
    spatial_dim: int,
    gas_temperature_K: float,
    collect_diagnostics: bool,
) -> tuple[list[np.ndarray], float, int, int]:
    output = [np.empty(plan.leaf_end_times_s.size, dtype=np.float64) for plan in plans]
    elapsed_total = 0.0
    point_total = 0
    call_total = 0
    for time_s, entries in sorted(_group_temperature_samples(plans).items()):
        temperatures, elapsed, point_count, call_count = _sample_temperature_group(
            compiled=compiled,
            entries=entries,
            time_s=time_s,
            spatial_dim=spatial_dim,
            gas_temperature_K=gas_temperature_K,
            collect_diagnostics=collect_diagnostics,
        )
        elapsed_total += elapsed
        point_total += point_count
        call_total += call_count
        for value, (plan_row, leaf_row, _point) in zip(
            temperatures, entries, strict=True
        ):
            output[plan_row][leaf_row] = float(value)
    _validate_plan_temperatures(plans, output, declared_source)
    return output, elapsed_total, point_total, call_total


def sample_plan_temperatures(
    *,
    config: StochasticMotionConfig,
    compiled: CompiledRuntimeBackend,
    plans: list[ParticleLeafPlan],
    spatial_dim: int,
    gas_temperature_K: float,
    collect_diagnostics: bool,
) -> tuple[list[np.ndarray], float, int, int]:
    declared_source, use_field = _temperature_source(config, compiled)
    if not use_field:
        return _configured_plan_temperatures(plans, gas_temperature_K), 0.0, 0, 0
    return _sample_field_plan_temperatures(
        compiled=compiled,
        plans=plans,
        declared_source=declared_source,
        spatial_dim=spatial_dim,
        gas_temperature_K=gas_temperature_K,
        collect_diagnostics=collect_diagnostics,
    )
