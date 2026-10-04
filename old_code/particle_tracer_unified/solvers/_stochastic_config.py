"""Configuration and cumulative diagnostics for stochastic motion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class StochasticMotionConfig:
    enabled: bool = False
    model: str = "underdamped_langevin"
    seed: int = 12345
    temperature_source: str = "field_T_then_gas"


def stochastic_motion_report(config: StochasticMotionConfig) -> dict[str, object]:
    return {
        "enabled": int(bool(config.enabled)),
        "model": str(config.model),
        "seed": int(config.seed),
        "temperature_source": str(config.temperature_source),
        "kick_event_count": 0,
        "kicked_particle_count": 0,
        "component_count": 0,
        "velocity_kick_rms_mps": 0.0,
        "last_velocity_kick_rms_mps": 0.0,
        "last_mean_sigma_v_mps": 0.0,
        "last_max_sigma_v_mps": 0.0,
        "last_mean_temperature_K": 0.0,
        "last_mean_tau_eff_s": 0.0,
    }


def _float_metric(values: Mapping[str, object], key: str) -> float:
    return float(np.asarray(values.get(key, 0.0)).item())


def _int_metric(values: Mapping[str, object], key: str) -> int:
    return int(np.asarray(values.get(key, 0)).item())


def merge_stochastic_motion_diagnostics(
    diagnostics: dict[str, object],
    config: StochasticMotionConfig,
    result: Mapping[str, object],
) -> None:
    if not bool(getattr(diagnostics, "debug", True)):
        return
    summary = diagnostics.setdefault(
        "stochastic_motion", stochastic_motion_report(config)
    )
    if not isinstance(summary, dict):
        summary = stochastic_motion_report(config)
        diagnostics["stochastic_motion"] = summary
    summary["enabled"] = int(bool(config.enabled))
    summary["model"] = str(config.model)
    summary["seed"] = int(config.seed)
    summary["temperature_source"] = str(config.temperature_source)
    if not bool(result.get("applied", False)):
        return
    particle_count = _int_metric(result, "particle_count")
    component_count = _int_metric(result, "component_count")
    sum_sq = _float_metric(result, "sum_sq")
    previous_components = _int_metric(summary, "component_count")
    previous_sum_sq = _float_metric(summary, "velocity_kick_sum_sq")
    total_components = int(previous_components + component_count)
    total_sum_sq = float(previous_sum_sq + sum_sq)
    summary["kick_event_count"] = _int_metric(summary, "kick_event_count") + 1
    summary["kicked_particle_count"] = _int_metric(
        summary, "kicked_particle_count"
    ) + int(particle_count)
    summary["component_count"] = int(total_components)
    summary["velocity_kick_sum_sq"] = float(total_sum_sq)
    summary["velocity_kick_rms_mps"] = (
        float(np.sqrt(total_sum_sq / total_components)) if total_components else 0.0
    )
    summary["last_velocity_kick_rms_mps"] = _float_metric(
        result, "rms_velocity_kick_mps"
    )
    summary["last_mean_sigma_v_mps"] = _float_metric(result, "mean_sigma_v_mps")
    summary["last_max_sigma_v_mps"] = _float_metric(result, "max_sigma_v_mps")
    summary["last_mean_temperature_K"] = _float_metric(result, "mean_temperature_K")
    summary["last_mean_tau_eff_s"] = _float_metric(result, "mean_tau_eff_s")
    diagnostics["field_sampling_s"] = _float_metric(
        diagnostics, "field_sampling_s"
    ) + _float_metric(result, "field_sampling_s")
    diagnostics["field_sample_point_count"] = _int_metric(
        diagnostics, "field_sample_point_count"
    ) + _int_metric(result, "field_sample_point_count")
    diagnostics["field_sample_call_count"] = _int_metric(
        diagnostics, "field_sample_call_count"
    ) + _int_metric(result, "field_sample_call_count")
