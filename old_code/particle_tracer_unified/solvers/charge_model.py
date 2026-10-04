"""Public particle-charging API."""

from __future__ import annotations

import numpy as np

from . import _charge_background as _background
from . import _charge_model_types as _types
from . import _charge_runtime as _runtime
from .plasma_background import PreparedPlasmaBackground, debye_length_m

AMU_KG = _types.AMU_KG
ELECTRON_MASS_KG = _types.ELECTRON_MASS_KG
EPS0_F_M = _types.EPS0_F_M
E_CHARGE_C = _types.E_CHARGE_C
ChargeModelConfig = _types.ChargeModelConfig
ChargeModelConfig.__module__ = __name__

charge_model_report = _types.charge_model_report
validate_charge_model_support = _background.validate_charge_model_support

_charge_response_regime = _types.charge_response_regime

merge_charge_model_diagnostics = _runtime.merge_charge_model_diagnostics
record_terminal_charge_replay = _runtime.record_terminal_charge_replay
finalize_charge_model_diagnostics = _runtime.finalize_charge_model_diagnostics


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
    plasma_background: PreparedPlasmaBackground | None = None,
    collect_diagnostics: bool = False,
) -> dict[str, object]:
    """Apply one charging update using physical-equivalent sphere diameters."""

    return _runtime.apply_charge_model_update(
        config=config,
        runtime=runtime,
        spatial_dim=spatial_dim,
        t_eval=t_eval,
        delta_t_s=delta_t_s,
        active_mask=active_mask,
        x=x,
        charge=charge,
        particle_diameter=particle_diameter,
        plasma_background=plasma_background,
        collect_diagnostics=collect_diagnostics,
        debye_length=debye_length_m,
        response_regime=_charge_response_regime,
    )


def advance_charge_strang_segment(
    *,
    config: ChargeModelConfig,
    runtime,
    spatial_dim: int,
    t_start_s: float,
    duration_s: float,
    x_start: np.ndarray,
    x_end: np.ndarray,
    charge_start: np.ndarray,
    particle_diameter: np.ndarray,
    plasma_background: PreparedPlasmaBackground | None = None,
) -> np.ndarray:
    """Advance charge using physical-equivalent sphere diameters."""

    duration = float(duration_s)
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("charge segment duration_s must be finite and non-negative")

    start = np.asarray(x_start, dtype=np.float64)
    end = np.asarray(x_end, dtype=np.float64)
    charge_out = np.asarray(charge_start, dtype=np.float64).copy()
    diameter = np.asarray(particle_diameter, dtype=np.float64)
    if start.ndim != 2 or end.shape != start.shape:
        raise ValueError("charge segment endpoints must be equally shaped 2D arrays")
    particle_count = int(start.shape[0])
    if charge_out.shape != (particle_count,) or diameter.shape != (particle_count,):
        raise ValueError("charge segment particle arrays must match endpoint rows")
    if duration == 0.0 or particle_count == 0 or not bool(config.enabled):
        return charge_out

    active = np.ones(particle_count, dtype=bool)
    half_duration = 0.5 * duration
    apply_charge_model_update(
        config=config,
        runtime=runtime,
        spatial_dim=int(spatial_dim),
        t_eval=float(t_start_s),
        delta_t_s=float(half_duration),
        active_mask=active,
        x=start,
        charge=charge_out,
        particle_diameter=diameter,
        plasma_background=plasma_background,
        collect_diagnostics=False,
    )
    apply_charge_model_update(
        config=config,
        runtime=runtime,
        spatial_dim=int(spatial_dim),
        t_eval=float(t_start_s) + duration,
        delta_t_s=float(half_duration),
        active_mask=active,
        x=end,
        charge=charge_out,
        particle_diameter=diameter,
        plasma_background=plasma_background,
        collect_diagnostics=False,
    )
    return charge_out


__all__ = (
    "AMU_KG",
    "ELECTRON_MASS_KG",
    "EPS0_F_M",
    "E_CHARGE_C",
    "ChargeModelConfig",
    "advance_charge_strang_segment",
    "apply_charge_model_update",
    "charge_model_report",
    "finalize_charge_model_diagnostics",
    "merge_charge_model_diagnostics",
    "record_terminal_charge_replay",
    "validate_charge_model_support",
)
