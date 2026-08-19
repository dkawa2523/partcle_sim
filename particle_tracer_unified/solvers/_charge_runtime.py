"""Apply particle-charge updates and accumulate their diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from ._charge_background import (
    ChargeBackground,
    ChargeUpdateBatch,
    complete_oml_background,
    prepare_charge_update_batch,
    resolve_charge_background,
)
from ._charge_model_types import E_CHARGE_C, ChargeModelConfig, charge_model_report
from ._charge_oml import oml_linearized_equilibrium, te_relaxation_equilibrium
from .plasma_background import PreparedPlasmaBackground


def _float_metric(values: Mapping[str, object], key: str) -> float:
    return float(np.asarray(values.get(key, 0.0)).item())


def _int_metric(values: Mapping[str, object], key: str) -> int:
    return int(np.asarray(values.get(key, 0)).item())


def charge_equilibrium(
    config: ChargeModelConfig,
    runtime,
    batch: ChargeUpdateBatch,
    background: ChargeBackground,
    *,
    t_eval: float,
    collect: bool,
    debye_length: Callable[..., np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ChargeBackground]:
    te_eV = background.electron_temperature_eV
    if not np.all(np.isfinite(te_eV)):
        raise ValueError(
            "solver.charge_model sampled non-finite electron temperature "
            "inside active particle support"
        )
    if str(config.mode) == "te_relaxation":
        q_eq, tau_q, phi = te_relaxation_equilibrium(config, batch.radius, te_eV)
        return q_eq, tau_q, phi, background
    resolved = complete_oml_background(
        config=config,
        runtime=runtime,
        batch=batch,
        t_eval=float(t_eval),
        background=background,
        collect=collect,
        debye_length=debye_length,
    )
    ne = np.asarray(resolved.electron_density_m3, dtype=np.float64)
    ni = np.asarray(resolved.ion_density_m3, dtype=np.float64)
    ti = np.asarray(resolved.ion_temperature_eV, dtype=np.float64)
    q_eq, tau_q, phi = oml_linearized_equilibrium(
        config,
        batch.radius,
        te_eV,
        ne,
        ni,
        ti,
        ion_mass_amu=resolved.ion_mass_amu,
        ion_charge_number=resolved.ion_charge_number,
    )
    return q_eq, tau_q, phi, resolved


def _positive_radius_ratio(radius: np.ndarray, debye: np.ndarray) -> np.ndarray:
    if not debye.size:
        return np.asarray([], dtype=np.float64)
    return np.divide(
        radius,
        debye,
        out=np.full_like(radius, np.nan, dtype=np.float64),
        where=np.isfinite(debye) & (debye > 0.0),
    )


def _mean_optional(values: np.ndarray | None) -> float:
    return float(np.mean(values)) if values is not None and values.size else 0.0


def charge_update_diagnostics(
    *,
    batch: ChargeUpdateBatch,
    background: ChargeBackground,
    updated: np.ndarray,
    equilibrium_charge: np.ndarray,
    relaxation_time: np.ndarray,
    floating_potential: np.ndarray,
    delta_t: float,
    response_regime: Callable[[float, np.ndarray], str],
) -> dict[str, object]:
    debye = (
        np.asarray(background.debye_length_m, dtype=np.float64)
        if background.debye_length_m is not None
        else np.asarray([], dtype=np.float64)
    )
    finite_debye = debye[np.isfinite(debye) & (debye > 0.0)]
    ratio = _positive_radius_ratio(batch.radius, debye)
    return {
        "applied": True,
        "particle_count": int(batch.indices.size),
        "background_source": background.source,
        "mean_Te_eV": float(np.mean(background.electron_temperature_eV)),
        "mean_ne_m3": _mean_optional(background.electron_density_m3),
        "mean_ni_m3": _mean_optional(background.ion_density_m3),
        "mean_Ti_eV": _mean_optional(background.ion_temperature_eV),
        "mean_debye_length_m": (
            float(np.mean(finite_debye)) if finite_debye.size else 0.0
        ),
        "mean_particle_radius_over_debye": (
            float(np.nanmean(ratio))
            if ratio.size and np.any(np.isfinite(ratio))
            else 0.0
        ),
        "mean_floating_potential_V": float(np.mean(floating_potential)),
        "mean_equilibrium_charge_e": float(np.mean(equilibrium_charge / E_CHARGE_C)),
        "mean_tau_q_s": float(np.mean(relaxation_time)),
        "charge_response_regime": response_regime(delta_t, relaxation_time),
        "mean_charge_C": float(np.mean(updated)),
        "mean_charge_e": float(np.mean(updated / E_CHARGE_C)),
    }


def apply_charge_model_update(
    *,
    config: ChargeModelConfig,
    runtime,
    spatial_dim: int,
    t_eval: float,
    delta_t_s: float,
    active_mask: np.ndarray,
    x: np.ndarray,
    charge: np.ndarray,
    particle_diameter: np.ndarray,
    plasma_background: PreparedPlasmaBackground | None,
    collect_diagnostics: bool,
    debye_length: Callable[..., np.ndarray],
    response_regime: Callable[[float, np.ndarray], str],
) -> dict[str, object]:
    if not bool(config.enabled):
        return {"applied": False}
    delta_t = float(delta_t_s)
    if not np.isfinite(delta_t) or delta_t < 0.0:
        raise ValueError(
            "solver.charge_model delta_t_s must be finite and non-negative"
        )
    if delta_t == 0.0:
        return {"applied": False}
    if int(spatial_dim) != 2:
        raise ValueError(
            "solver.charge_model currently supports only 2D; "
            "3D support is planned separately"
        )
    batch = prepare_charge_update_batch(active_mask, x, charge, particle_diameter)
    if batch is None:
        return {"applied": False, "particle_count": 0}
    collect = bool(collect_diagnostics)
    background = resolve_charge_background(
        config=config,
        runtime=runtime,
        batch=batch,
        t_eval=float(t_eval),
        plasma_background=plasma_background,
        collect=collect,
    )
    q_eq, tau_q, phi, background = charge_equilibrium(
        config,
        runtime,
        batch,
        background,
        t_eval=float(t_eval),
        collect=collect,
        debye_length=debye_length,
    )
    if np.any(~np.isfinite(tau_q) | (tau_q <= 0.0)):
        raise ValueError("solver.charge_model produced a non-positive relaxation time")
    decay = np.exp(-delta_t / tau_q)
    new_charge = q_eq + (batch.old_charge - q_eq) * decay
    if np.any(~np.isfinite(new_charge)):
        rows = np.flatnonzero(~np.isfinite(new_charge))[:12].tolist()
        raise ValueError(
            f"solver.charge_model produced non-finite charge at active rows {rows}"
        )
    charge[batch.indices] = new_charge
    if not collect:
        return {"applied": True}
    return charge_update_diagnostics(
        batch=batch,
        background=background,
        updated=np.asarray(charge[batch.indices], dtype=np.float64),
        equilibrium_charge=q_eq,
        relaxation_time=tau_q,
        floating_potential=phi,
        delta_t=delta_t,
        response_regime=response_regime,
    )


def merge_charge_model_diagnostics(
    diagnostics: dict[str, object],
    config: ChargeModelConfig,
    result: Mapping[str, object],
) -> None:
    if not bool(getattr(diagnostics, "debug", True)):
        return
    summary = diagnostics.setdefault("charge_model", charge_model_report(config))
    if not isinstance(summary, dict):
        summary = charge_model_report(config)
        diagnostics["charge_model"] = summary
    summary["enabled"] = int(bool(config.enabled))
    summary["mode"] = str(config.mode)
    summary["background_source"] = str(config.background_source)
    summary["ion_velocity_model"] = str(config.ion_velocity_model)
    if not bool(result.get("applied", False)):
        return
    particle_count = _int_metric(result, "particle_count")
    summary["update_event_count"] = _int_metric(summary, "update_event_count") + 1
    summary["updated_particle_count"] = _int_metric(
        summary, "updated_particle_count"
    ) + int(particle_count)
    summary["last_updated_particle_count"] = int(particle_count)
    summary["last_mean_Te_eV"] = _float_metric(result, "mean_Te_eV")
    summary["last_mean_ne_m3"] = _float_metric(result, "mean_ne_m3")
    summary["last_mean_ni_m3"] = _float_metric(result, "mean_ni_m3")
    summary["last_mean_Ti_eV"] = _float_metric(result, "mean_Ti_eV")
    summary["last_mean_debye_length_m"] = _float_metric(result, "mean_debye_length_m")
    summary["last_mean_particle_radius_over_debye"] = _float_metric(
        result, "mean_particle_radius_over_debye"
    )
    summary["last_mean_floating_potential_V"] = _float_metric(
        result, "mean_floating_potential_V"
    )
    summary["last_mean_equilibrium_charge_e"] = _float_metric(
        result, "mean_equilibrium_charge_e"
    )
    summary["last_mean_tau_q_s"] = _float_metric(result, "mean_tau_q_s")
    summary["last_charge_response_regime"] = str(
        result.get("charge_response_regime", "unknown")
    )
    summary["last_mean_charge_C"] = _float_metric(result, "mean_charge_C")
    summary["last_mean_charge_e"] = _float_metric(result, "mean_charge_e")


def record_terminal_charge_replay(
    diagnostics: dict[str, object],
    config: ChargeModelConfig,
    *,
    age_s: float,
) -> None:
    if not bool(getattr(diagnostics, "debug", True)):
        return
    summary = diagnostics.setdefault("charge_model", charge_model_report(config))
    if not isinstance(summary, dict):
        summary = charge_model_report(config)
        diagnostics["charge_model"] = summary
    age = max(float(age_s), 0.0)
    summary["terminal_hit_replay_count"] = (
        _int_metric(summary, "terminal_hit_replay_count") + 1
    )
    summary["terminal_hit_replay_age_total_s"] = (
        _float_metric(summary, "terminal_hit_replay_age_total_s") + age
    )
    summary["terminal_hit_replay_age_max_s"] = max(
        _float_metric(summary, "terminal_hit_replay_age_max_s"),
        age,
    )


def finalize_charge_model_diagnostics(
    diagnostics: dict[str, object],
    config: ChargeModelConfig,
    charge: np.ndarray,
) -> None:
    if not bool(getattr(diagnostics, "debug", True)):
        return
    summary = diagnostics.setdefault("charge_model", charge_model_report(config))
    if not isinstance(summary, dict):
        summary = charge_model_report(config)
        diagnostics["charge_model"] = summary
    values = np.asarray(charge, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    summary["final_min_charge_C"] = float(np.min(finite))
    summary["final_mean_charge_C"] = float(np.mean(finite))
    summary["final_max_charge_C"] = float(np.max(finite))
    charge_e = finite / E_CHARGE_C
    summary["final_min_charge_e"] = float(np.min(charge_e))
    summary["final_mean_charge_e"] = float(np.mean(charge_e))
    summary["final_max_charge_e"] = float(np.max(charge_e))


__all__ = (
    "apply_charge_model_update",
    "finalize_charge_model_diagnostics",
    "merge_charge_model_diagnostics",
    "record_terminal_charge_replay",
)
