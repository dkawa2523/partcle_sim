"""Side-effect-free release-state validation for gas-drag applicability."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core.field_backend import ProviderSamplingBackend
from .core.field_sampling import choose_velocity_quantity_names
from .domain import FieldRequest
from .solvers.drag_regime import (
    DragRegimeDecision,
    classify_drag_regime,
    gas_mean_free_path_m,
    particle_reynolds_number,
    relative_knudsen_number,
    relative_mach_number,
)

_GAS_FIELD_ALIASES = {
    "density_kgm3": ("rho_g", "rho", "gas_density"),
    "dynamic_viscosity_Pas": ("mu", "dynamic_viscosity", "gas_mu"),
    "temperature_K": ("T", "temperature", "gas_temperature"),
    "sound_speed_mps": ("sound_speed", "speed_of_sound", "c_sound"),
}
_AMU_KG = 1.66053906660e-27


@dataclass(frozen=True)
class DragRegimeFinding:
    code: str
    message: str
    severity: str
    context: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _ReleaseSamples:
    active_indices: np.ndarray
    times: np.ndarray
    clean_support: np.ndarray
    reynolds: np.ndarray
    knudsen: np.ndarray
    mach: np.ndarray


@dataclass(frozen=True, slots=True)
class _RegimeAssessment:
    decisions: tuple[DragRegimeDecision | None, ...]
    assessable: np.ndarray
    missing: np.ndarray
    error_reasons: Mapping[str, int]
    warning_reasons: Mapping[str, int]
    error_count: int
    warning_count: int


def _finite_range(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "finite_count": int(finite.size),
        "min": float(np.min(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
    }


def _base_report(model: str) -> dict[str, Any]:
    return {
        "scope": "initial_release_state",
        "model": model,
        "dynamic_history_assessed": False,
    }


def _not_applicable_report(base: Mapping[str, Any], reason: str) -> dict[str, Any]:
    return {
        **base,
        "passed": True,
        "applicable": False,
        "reason": reason,
        "integrated_particle_count": 0,
        "assessed_particle_count": 0,
        "error_count": 0,
        "warning_count": 0,
        "metrics": {},
    }


def _active_release_indices(runtime: Any) -> np.ndarray:
    release_time = np.asarray(runtime.particles.release_time, dtype=np.float64)
    t_end = float(getattr(runtime.plan, "t_end", 0.0))
    return np.flatnonzero(
        np.isfinite(release_time) & (release_time >= 0.0) & (release_time < t_end)
    )


def _selected_gas_fields(field: Any) -> dict[str, str]:
    quantity_names = set(map(str, getattr(field, "quantities", {})))
    return {
        name: next((alias for alias in aliases if alias in quantity_names), "")
        for name, aliases in _GAS_FIELD_ALIASES.items()
    }


def _initial_gas_values(runtime: Any, count: int) -> dict[str, np.ndarray]:
    return {
        "density_kgm3": np.full(
            count,
            float(runtime.gas.density_kgm3),
            dtype=np.float64,
        ),
        "dynamic_viscosity_Pas": np.full(
            count,
            float(runtime.gas.dynamic_viscosity_Pas),
            dtype=np.float64,
        ),
        "temperature_K": np.full(
            count,
            float(runtime.gas.temperature),
            dtype=np.float64,
        ),
        "sound_speed_mps": np.full(count, np.nan, dtype=np.float64),
    }


def _sample_groups(
    field: Any,
    requested: tuple[str, ...],
    times: np.ndarray,
    count: int,
    spatial_dim: int,
) -> list[tuple[float, np.ndarray]]:
    series = [field.quantities[name] for name in requested]
    steady = all(
        np.asarray(item.data).ndim == spatial_dim
        or np.asarray(item.times, dtype=np.float64).size <= 1
        for item in series
    )
    if steady:
        return [(0.0, np.arange(count, dtype=np.int64))]
    return [
        (float(time_s), np.flatnonzero(times == time_s)) for time_s in np.unique(times)
    ]


def _sample_release_environment(
    runtime: Any,
    active_indices: np.ndarray,
    velocity_names: tuple[str, ...],
    gas_field_names: Mapping[str, str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    particles = runtime.particles
    spatial_dim = int(runtime.spatial_dim)
    count = int(active_indices.size)
    positions = np.asarray(
        particles.position[active_indices, :spatial_dim],
        dtype=np.float64,
    )
    times = np.asarray(particles.release_time[active_indices], dtype=np.float64)
    flow = np.full((count, spatial_dim), np.nan, dtype=np.float64)
    clean_support = np.zeros(count, dtype=bool)
    gas_values = _initial_gas_values(runtime, count)
    requested = tuple(
        dict.fromkeys(
            (*velocity_names, *(name for name in gas_field_names.values() if name))
        )
    )
    if not requested:
        return flow, clean_support, gas_values
    field_provider = runtime.field_provider
    field = field_provider.field
    backend = ProviderSamplingBackend(field_provider)
    for time_s, local in _sample_groups(
        field,
        requested,
        times,
        count,
        spatial_dim,
    ):
        sampled = backend.sample(positions[local], time_s, FieldRequest(requested))
        clean_support[local] = np.asarray(sampled.supported, dtype=bool)
        for axis, name in enumerate(velocity_names):
            flow[local, axis] = np.asarray(sampled.values[name], dtype=np.float64)
        for gas_name, field_name in gas_field_names.items():
            if field_name:
                # Explicit field values are authoritative, including invalid values.
                gas_values[gas_name][local] = np.asarray(
                    sampled.values[field_name],
                    dtype=np.float64,
                )
    return flow, clean_support, gas_values


def _release_samples(runtime: Any, active_indices: np.ndarray) -> _ReleaseSamples:
    particles = runtime.particles
    spatial_dim = int(runtime.spatial_dim)
    field = getattr(runtime.field_provider, "field", None)
    velocity_names = tuple(choose_velocity_quantity_names(field, spatial_dim))
    gas_field_names = _selected_gas_fields(field)
    flow, clean_support, gas_values = _sample_release_environment(
        runtime,
        active_indices,
        velocity_names,
        gas_field_names,
    )
    velocities = np.asarray(
        particles.velocity[active_indices, :spatial_dim],
        dtype=np.float64,
    )
    diameters = np.asarray(particles.diameter[active_indices], dtype=np.float64)
    times = np.asarray(particles.release_time[active_indices], dtype=np.float64)
    slip_speed = np.linalg.norm(velocities - flow, axis=1)
    density = gas_values["density_kgm3"]
    viscosity = gas_values["dynamic_viscosity_Pas"]
    reynolds = particle_reynolds_number(slip_speed, diameters, density, viscosity)
    molecular_mass = np.full(
        active_indices.size,
        float(runtime.gas.molecular_mass_amu) * _AMU_KG,
        dtype=np.float64,
    )
    mean_free_path = gas_mean_free_path_m(
        viscosity,
        density,
        gas_values["temperature_K"],
        molecular_mass,
    )
    return _ReleaseSamples(
        active_indices=active_indices,
        times=times,
        clean_support=clean_support,
        reynolds=reynolds,
        knudsen=relative_knudsen_number(mean_free_path, diameters),
        mach=relative_mach_number(slip_speed, gas_values["sound_speed_mps"]),
    )


def _increment_reasons(target: dict[str, int], reasons: tuple[str, ...]) -> None:
    for reason in reasons:
        target[reason] = target.get(reason, 0) + 1


def _assess_regimes(model: str, samples: _ReleaseSamples) -> _RegimeAssessment:
    assessable = (
        samples.clean_support
        & np.isfinite(samples.reynolds)
        & np.isfinite(samples.knudsen)
    )
    missing = samples.clean_support & ~assessable
    decisions: list[DragRegimeDecision | None] = []
    error_reasons: dict[str, int] = {}
    warning_reasons: dict[str, int] = {}
    for index in range(samples.active_indices.size):
        if not assessable[index]:
            decisions.append(None)
            continue
        decision = classify_drag_regime(
            model,
            reynolds=float(samples.reynolds[index]),
            knudsen=float(samples.knudsen[index]),
            relative_mach=float(samples.mach[index]),
        )
        decisions.append(decision)
        _increment_reasons(error_reasons, decision.errors)
        _increment_reasons(warning_reasons, decision.warnings)
    missing_count = int(np.count_nonzero(missing))
    if missing_count:
        error_reasons["regime_inputs_unavailable"] = missing_count
    return _RegimeAssessment(
        decisions=tuple(decisions),
        assessable=assessable,
        missing=missing,
        error_reasons=error_reasons,
        warning_reasons=warning_reasons,
        error_count=missing_count
        + sum(decision is not None and bool(decision.errors) for decision in decisions),
        warning_count=sum(
            decision is not None and bool(decision.warnings) for decision in decisions
        ),
    )


def _findings(
    model: str,
    assessment: _RegimeAssessment,
) -> tuple[DragRegimeFinding, ...]:
    findings: list[DragRegimeFinding] = []
    if assessment.error_reasons:
        findings.append(
            DragRegimeFinding(
                code="physics.drag.regime",
                message=(
                    "The declared drag law is not valid or cannot be verified "
                    "at one or more clean release states"
                ),
                severity="error",
                context={
                    "model": model,
                    "scope": "initial_release_state",
                    "reason_counts": dict(sorted(assessment.error_reasons.items())),
                },
            )
        )
    if assessment.warning_reasons:
        findings.append(
            DragRegimeFinding(
                code="physics.drag.regime.transition",
                message=(
                    "The declared drag law starts in a transition range and "
                    "requires model review"
                ),
                severity="warning",
                context={
                    "model": model,
                    "scope": "initial_release_state",
                    "reason_counts": dict(sorted(assessment.warning_reasons.items())),
                },
            )
        )
    return tuple(findings)


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _violation_rows(
    runtime: Any,
    samples: _ReleaseSamples,
    assessment: _RegimeAssessment,
    include_violations: bool,
) -> list[dict[str, Any]]:
    if not include_violations:
        return []
    rows: list[dict[str, Any]] = []
    for local_index, decision in enumerate(assessment.decisions):
        errors = (
            ("regime_inputs_unavailable",)
            if bool(assessment.missing[local_index])
            else (() if decision is None else decision.errors)
        )
        warnings = () if decision is None else decision.warnings
        if not errors and not warnings:
            continue
        particle_index = int(samples.active_indices[local_index])
        rows.append(
            {
                "particle_id": int(runtime.particles.particle_id[particle_index]),
                "release_time_s": float(samples.times[local_index]),
                "particle_reynolds": _finite_or_none(samples.reynolds[local_index]),
                "relative_knudsen": _finite_or_none(samples.knudsen[local_index]),
                "relative_mach": _finite_or_none(samples.mach[local_index]),
                "errors": list(errors),
                "warnings": list(warnings),
            }
        )
    return rows


def _report(
    base: Mapping[str, Any],
    samples: _ReleaseSamples,
    assessment: _RegimeAssessment,
    violations: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        **base,
        "passed": assessment.error_count == 0,
        "applicable": True,
        "integrated_particle_count": int(samples.active_indices.size),
        "assessed_particle_count": int(np.count_nonzero(assessment.assessable)),
        "unassessed_clean_particle_count": int(np.count_nonzero(assessment.missing)),
        "relative_mach_assessed_count": int(
            np.count_nonzero(samples.clean_support & np.isfinite(samples.mach))
        ),
        "error_count": assessment.error_count,
        "warning_count": assessment.warning_count,
        "error_reason_counts": dict(sorted(assessment.error_reasons.items())),
        "warning_reason_counts": dict(sorted(assessment.warning_reasons.items())),
        "metrics": {
            "particle_reynolds": _finite_range(samples.reynolds[samples.clean_support]),
            "relative_knudsen": _finite_range(samples.knudsen[samples.clean_support]),
            "relative_mach": _finite_range(samples.mach[samples.clean_support]),
        },
        "relative_mach_note": (
            "evaluated only when the field artifact explicitly supplies "
            "sound_speed, speed_of_sound, or c_sound"
        ),
        "violations": violations,
    }


def initial_drag_regime_report(
    runtime: Any,
    *,
    include_violations: bool,
) -> tuple[dict[str, Any], tuple[DragRegimeFinding, ...]]:
    """Assess the declared drag law at integrated-particle release states."""

    model = str(getattr(getattr(runtime, "plan", None), "drag_model_name", "none"))
    base = _base_report(model)
    if model == "none":
        return _not_applicable_report(base, "drag_disabled"), ()
    active_indices = _active_release_indices(runtime)
    if active_indices.size == 0:
        reason = "no_particle_integrates_before_t_end"
        return _not_applicable_report(base, reason), ()
    samples = _release_samples(runtime, active_indices)
    assessment = _assess_regimes(model, samples)
    violations = _violation_rows(
        runtime,
        samples,
        assessment,
        include_violations,
    )
    return _report(base, samples, assessment, violations), _findings(model, assessment)


__all__ = ("DragRegimeFinding", "initial_drag_regime_report")
