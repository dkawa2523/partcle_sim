from __future__ import annotations

import numpy as np
from numba import njit

from .drag_models import (
    _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA as _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
)
from .drag_models import DRAG_MODEL_EPSTEIN as DRAG_MODEL_EPSTEIN
from .drag_models import DRAG_MODEL_NONE as DRAG_MODEL_NONE
from .drag_models import (
    DRAG_MODEL_SCHILLER_NAUMANN as DRAG_MODEL_SCHILLER_NAUMANN,
)
from .drag_models import DRAG_MODEL_STOKES as DRAG_MODEL_STOKES
from .drag_models import (
    DRAG_MODEL_STOKES_CUNNINGHAM as DRAG_MODEL_STOKES_CUNNINGHAM,
)
from .drag_models import _is_positive_finite as _is_positive_finite
from .drag_models import (
    cunningham_slip_correction as cunningham_slip_correction,
)
from .drag_models import drag_model_mode_from_name as drag_model_mode_from_name
from .drag_models import drag_model_name_from_mode as drag_model_name_from_mode
from .drag_models import (
    effective_tau_from_drag_model as effective_tau_from_drag_model,
)
from .drag_models import (
    effective_tau_from_slip_speed as effective_tau_from_slip_speed,
)
from .drag_models import epstein_relaxation_time as epstein_relaxation_time
from .drag_models import (
    schiller_naumann_drag_correction as schiller_naumann_drag_correction,
)
from .drag_models import stokes_relaxation_time as stokes_relaxation_time

# ETD2 step doubling estimates the local defect without introducing a
# dimensional, case-specific absolute tolerance.  The relative part scales
# with the state change over the candidate leaf; the ULP allowance only
# absorbs floating-point roundoff at the represented state magnitude.
ETD2_LOCAL_ERROR_RTOL = 1.0e-5
ETD2_LOCAL_ERROR_ULPS = 64.0
ETD2_STEP_DOUBLING_COARSE_ERROR_FACTOR = 4.0 / 3.0


@njit(cache=True, inline="always")
def _maximum_substeps(adaptive_max_splits):
    maximum = 1
    splits = int(max(0, adaptive_max_splits))
    for _ in range(splits):
        maximum *= 2
    return maximum


@njit(cache=True)
def uniform_substep_schedule(
    duration_s,
    end_time_s,
    adaptive_max_splits,
    minimum_substeps,
):
    """Return the uniform schedule used by LTE-driven ETD2 refinement."""

    duration = max(float(duration_s), 0.0)
    count = max(1, int(minimum_substeps))
    maximum = _maximum_substeps(adaptive_max_splits)
    if count > maximum:
        count = maximum
    return count, duration / float(count), float(end_time_s) - duration


@njit(cache=True, inline="always")
def _state_ulp_allowance(value_start, value_full, value_refined):
    spacing = max(
        abs(np.spacing(value_start)),
        abs(np.spacing(value_full)),
        abs(np.spacing(value_refined)),
    )
    return ETD2_LOCAL_ERROR_ULPS * spacing


@njit(cache=True, inline="always")
def etd2_position_error_exceeds_tolerance(
    position_start,
    position_full,
    position_refined,
    velocity_start,
    velocity_full,
    velocity_refined,
    duration_s,
):
    """Compare one position component using a translation-safe local scale."""

    if not (
        np.isfinite(position_start)
        and np.isfinite(position_full)
        and np.isfinite(position_refined)
        and np.isfinite(velocity_start)
        and np.isfinite(velocity_full)
        and np.isfinite(velocity_refined)
    ):
        return True
    duration = max(float(duration_s), 0.0)
    motion_scale = max(
        abs(position_full - position_start),
        abs(position_refined - position_start),
        duration * max(abs(velocity_start), abs(velocity_full), abs(velocity_refined)),
    )
    allowance = ETD2_LOCAL_ERROR_RTOL * motion_scale + _state_ulp_allowance(
        position_start, position_full, position_refined
    )
    estimated_coarse_error = ETD2_STEP_DOUBLING_COARSE_ERROR_FACTOR * abs(
        position_full - position_refined
    )
    return estimated_coarse_error > allowance


@njit(cache=True, inline="always")
def etd2_velocity_error_exceeds_tolerance(
    velocity_start,
    velocity_full,
    velocity_refined,
):
    """Compare one velocity component with relative and ULP weighting."""

    if not (
        np.isfinite(velocity_start)
        and np.isfinite(velocity_full)
        and np.isfinite(velocity_refined)
    ):
        return True
    velocity_scale = max(
        abs(velocity_start),
        abs(velocity_full),
        abs(velocity_refined),
        abs(velocity_full - velocity_start),
        abs(velocity_refined - velocity_start),
    )
    allowance = ETD2_LOCAL_ERROR_RTOL * velocity_scale + _state_ulp_allowance(
        velocity_start, velocity_full, velocity_refined
    )
    estimated_coarse_error = ETD2_STEP_DOUBLING_COARSE_ERROR_FACTOR * abs(
        velocity_full - velocity_refined
    )
    return estimated_coarse_error > allowance


@njit(cache=True, inline="always")
def doubled_substep_count(current_substeps, adaptive_max_splits):
    """Double a uniform schedule without exceeding its configured split cap."""

    current = max(1, int(current_substeps))
    maximum = _maximum_substeps(adaptive_max_splits)
    return min(maximum, 2 * current)


@njit(cache=True, inline="always")
def etd2_stage_schedule(substep_start_time_s, substep_duration_s):
    """Return canonical midpoint time and predictor/corrector durations."""

    duration = max(float(substep_duration_s), 0.0)
    predictor_duration = 0.5 * duration
    return (
        float(substep_start_time_s) + predictor_duration,
        predictor_duration,
        duration,
    )


@njit(cache=True, inline="always")
def compose_stage_acceleration_2d(
    body_ax,
    body_ay,
    force_ax,
    force_ay,
    gas_density_kgm3,
    particle_density_kgm3,
    gravity_buoyancy_enabled,
    inertia_factor,
):
    """Compose 2D body, buoyancy, and canonical pipeline acceleration."""

    gravity_factor = 1.0
    if gravity_buoyancy_enabled != 0 and particle_density_kgm3 > 0.0:
        gravity_factor = 1.0 - gas_density_kgm3 / particle_density_kgm3
    inverse_inertia = 1.0 / inertia_factor
    return (
        (gravity_factor * body_ax + force_ax) * inverse_inertia,
        (gravity_factor * body_ay + force_ay) * inverse_inertia,
    )


@njit(cache=True, inline="always")
def compose_stage_acceleration_3d(
    body_ax,
    body_ay,
    body_az,
    force_ax,
    force_ay,
    force_az,
    gas_density_kgm3,
    particle_density_kgm3,
    gravity_buoyancy_enabled,
    inertia_factor,
):
    """Compose 3D body, buoyancy, and canonical pipeline acceleration."""

    gravity_factor = 1.0
    if gravity_buoyancy_enabled != 0 and particle_density_kgm3 > 0.0:
        gravity_factor = 1.0 - gas_density_kgm3 / particle_density_kgm3
    inverse_inertia = 1.0 / inertia_factor
    return (
        (gravity_factor * body_ax + force_ax) * inverse_inertia,
        (gravity_factor * body_ay + force_ay) * inverse_inertia,
        (gravity_factor * body_az + force_az) * inverse_inertia,
    )


@njit(cache=True)
def advance_component(v0, target, body_accel, tau_eff, dt):
    """Integrate one constant-coefficient predictor or unchanged ETD stage."""

    if dt <= 0.0:
        return 0.0, v0
    if np.isinf(tau_eff) and tau_eff > 0.0:
        # Explicit drag_model=none: exact constant-acceleration ballistic step.
        return v0 * dt + 0.5 * body_accel * dt * dt, v0 + body_accel * dt
    if not np.isfinite(tau_eff) or tau_eff <= 0.0:
        return np.nan, np.nan

    ratio = dt / tau_eff
    one_minus_decay = -np.expm1(-ratio)
    decay = 1.0 - one_minus_decay
    # f = dt - tau * (1-exp(-dt/tau)).  The series avoids cancellation
    # for ballistic-scale tau values.
    if abs(ratio) < 1.0e-4:
        ratio2 = ratio * ratio
        f = tau_eff * (
            0.5 * ratio2
            - ratio2 * ratio / 6.0
            + ratio2 * ratio2 / 24.0
            - ratio2 * ratio2 * ratio / 120.0
        )
    else:
        f = dt - tau_eff * one_minus_decay
    v1 = target + (v0 - target) * decay + body_accel * tau_eff * one_minus_decay
    x_delta = v0 * dt + (target - v0) * f + body_accel * tau_eff * f
    return x_delta, v1


@njit(cache=True, inline="always")
def _positive_relaxation_time_or_infinity(value):
    return (np.isfinite(value) and value > 0.0) or (np.isinf(value) and value > 0.0)


@njit(cache=True, inline="always")
def _affine_stage_relaxation_time(tau_start, tau_mid, stage_fraction):
    if not _positive_relaxation_time_or_infinity(
        tau_start
    ) or not _positive_relaxation_time_or_infinity(tau_mid):
        return np.nan
    if tau_start == tau_mid or stage_fraction == 1.0:
        return tau_mid
    inverse_tau_start = 0.0 if np.isinf(tau_start) else 1.0 / tau_start
    inverse_tau_mid = 0.0 if np.isinf(tau_mid) else 1.0 / tau_mid
    inverse_tau = (
        1.0 - stage_fraction
    ) * inverse_tau_start + stage_fraction * inverse_tau_mid
    return np.inf if inverse_tau == 0.0 else 1.0 / inverse_tau


@njit(cache=True, inline="always")
def _affine_etd_weights(ratio):
    one_minus_decay = -np.expm1(-ratio)
    if abs(ratio) < 5.0e-2:
        ratio2 = ratio * ratio
        phi1 = (
            1.0
            - 0.5 * ratio
            + ratio2 / 6.0
            - ratio2 * ratio / 24.0
            + ratio2 * ratio2 / 120.0
            - ratio2 * ratio2 * ratio / 720.0
        )
        slope_velocity_weight = ratio * (
            0.5
            - ratio / 6.0
            + ratio2 / 24.0
            - ratio2 * ratio / 120.0
            + ratio2 * ratio2 / 720.0
        )
        slope_position_weight = ratio * (
            1.0 / 6.0
            - ratio / 24.0
            + ratio2 / 120.0
            - ratio2 * ratio / 720.0
            + ratio2 * ratio2 / 5040.0
        )
        return (
            1.0 - one_minus_decay,
            one_minus_decay,
            phi1,
            slope_velocity_weight,
            slope_position_weight,
        )
    inverse_ratio = 1.0 / ratio
    return (
        1.0 - one_minus_decay,
        one_minus_decay,
        one_minus_decay * inverse_ratio,
        1.0 - one_minus_decay * inverse_ratio,
        0.5 - inverse_ratio + one_minus_decay * inverse_ratio * inverse_ratio,
    )


@njit(cache=True, inline="always")
def advance_affine_stage_component(
    v0,
    target_start,
    target_mid,
    accel_start,
    accel_mid,
    tau_start,
    tau_mid,
    stage_fraction,
    dt,
):
    """Advance with stage-local relaxation and linearly reconstructed forcing.

    Start and midpoint samples reconstruct the forcing at ``stage_fraction``
    of the leaf.  Integrating that affine forcing analytically avoids the
    stiff-limit order reduction of a midpoint-frozen target.  Unchanged
    coefficients retain the original ETD operation sequence exactly.
    """

    if target_start == target_mid and accel_start == accel_mid and tau_start == tau_mid:
        return advance_component(v0, target_mid, accel_mid, tau_mid, dt)
    if dt <= 0.0:
        return 0.0, v0
    tau_effective = _affine_stage_relaxation_time(
        tau_start,
        tau_mid,
        stage_fraction,
    )
    if np.isnan(tau_effective):
        return np.nan, np.nan
    forcing_delta_factor = 2.0 * stage_fraction
    if np.isinf(tau_effective):
        acceleration_delta = forcing_delta_factor * (accel_mid - accel_start)
        velocity = v0 + dt * (accel_start + 0.5 * acceleration_delta)
        displacement = dt * v0 + dt * dt * (
            0.5 * accel_start + acceleration_delta / 6.0
        )
        return displacement, velocity
    ratio = dt / tau_effective
    (
        decay,
        one_minus_decay,
        phi1,
        slope_velocity_weight,
        slope_position_weight,
    ) = _affine_etd_weights(ratio)

    equilibrium_start = target_start + tau_effective * accel_start
    equilibrium_mid = target_mid + tau_effective * accel_mid
    equilibrium_delta = forcing_delta_factor * (equilibrium_mid - equilibrium_start)
    velocity = (
        decay * v0
        + one_minus_decay * equilibrium_start
        + slope_velocity_weight * equilibrium_delta
    )
    displacement = dt * (
        phi1 * v0
        + slope_velocity_weight * equilibrium_start
        + slope_position_weight * equilibrium_delta
    )
    return displacement, velocity


@njit(cache=True)
def advance_state_2d(x0, y0, vx0, vy0, targetx, targety, bax, bay, tau_eff, dt):
    dx, vx1 = advance_component(vx0, targetx, bax, tau_eff, dt)
    dy, vy1 = advance_component(vy0, targety, bay, tau_eff, dt)
    return x0 + dx, y0 + dy, vx1, vy1


@njit(cache=True)
def advance_state_3d(
    x0, y0, z0, vx0, vy0, vz0, targetx, targety, targetz, bax, bay, baz, tau_eff, dt
):
    dx, vx1 = advance_component(vx0, targetx, bax, tau_eff, dt)
    dy, vy1 = advance_component(vy0, targety, bay, tau_eff, dt)
    dz, vz1 = advance_component(vz0, targetz, baz, tau_eff, dt)
    return x0 + dx, y0 + dy, z0 + dz, vx1, vy1, vz1
