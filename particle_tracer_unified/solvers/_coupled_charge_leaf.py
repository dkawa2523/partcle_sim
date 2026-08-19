"""One deterministic ETD leaf with charge and electric force coupled in time."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ._segment_motion_contracts import SegmentMotionRequest
from ._segment_stage_dynamics import (
    _advance_with_affine_stage_dynamics,
    _advance_with_stage_dynamics,
    _sample_stage_dynamics,
    _stage_sampling_needs,
    _StageDynamics,
    _StageDynamicsContext,
)
from .charge_model import ChargeModelConfig, apply_charge_model_update
from .integrator_common import (
    DRAG_MODEL_NONE,
    etd2_position_error_exceeds_tolerance,
    etd2_velocity_error_exceeds_tolerance,
)
from .plasma_background import PreparedPlasmaBackground


@dataclass(frozen=True, slots=True)
class CoupledChargeLeafContext:
    motion: _StageDynamicsContext
    charge_config: ChargeModelConfig
    runtime: object
    plasma_background: PreparedPlasmaBackground | None
    physical_diameter_m: float


@dataclass(frozen=True, slots=True)
class CoupledChargeLeafState:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    charge_C: float
    sample_positions_m: np.ndarray


@dataclass(frozen=True, slots=True)
class CoupledChargeEmbeddedStep:
    full: CoupledChargeLeafState
    refined_mid: CoupledChargeLeafState
    refined_end: CoupledChargeLeafState
    refinement_required: bool


def coupled_charge_leaf_context(
    request: SegmentMotionRequest,
    *,
    config: ChargeModelConfig,
    runtime: object,
    plasma_background: PreparedPlasmaBackground | None,
    physical_diameter_m: float,
) -> CoupledChargeLeafContext:
    return CoupledChargeLeafContext(
        motion=_StageDynamicsContext(
            spatial_dim=int(request.spatial_dim),
            compiled=request.backend,
            body_acceleration=np.asarray(
                request.body_acceleration_mps2, dtype=np.float64
            )[: request.spatial_dim],
            tau_stokes_s=float(request.tau_stokes_s),
            particle_diameter_m=float(request.particle_diameter_m),
            particle_density_kgm3=float(request.particle_density_kgm3),
            particle_mass_kg=float(request.particle_mass_kg),
            dep_particle_rel_permittivity=float(request.dep_particle_rel_permittivity),
            thermophoretic_coefficient=float(request.thermophoretic_coefficient),
            fallback_density_kgm3=float(request.gas_density_kgm3),
            fallback_viscosity_Pas=float(request.gas_dynamic_viscosity_Pas),
            fallback_temperature_K=float(request.gas_temperature_K),
            gas_molecular_mass_kg=float(request.gas_molecular_mass_kg),
            drag_model_mode=int(request.drag_model_mode),
            electric_q_over_m_Ckg=0.0,
            force_runtime=request.force_runtime,
            sampling_needs=_stage_sampling_needs(
                int(request.drag_model_mode), request.force_runtime
            ),
        ),
        charge_config=config,
        runtime=runtime,
        plasma_background=plasma_background,
        physical_diameter_m=float(physical_diameter_m),
    )


def _advance_charge(
    context: CoupledChargeLeafContext,
    *,
    charge_C: float,
    position_m: np.ndarray,
    time_s: float,
    duration_s: float,
) -> float:
    charge = np.asarray([float(charge_C)], dtype=np.float64)
    apply_charge_model_update(
        config=context.charge_config,
        runtime=context.runtime,
        spatial_dim=int(context.motion.spatial_dim),
        t_eval=float(time_s),
        delta_t_s=float(duration_s),
        active_mask=np.ones(1, dtype=bool),
        x=np.asarray(position_m, dtype=np.float64).reshape(1, -1),
        charge=charge,
        particle_diameter=np.asarray(
            [float(context.physical_diameter_m)], dtype=np.float64
        ),
        plasma_background=context.plasma_background,
        collect_diagnostics=False,
    )
    return float(charge[0])


def _sample_motion(
    context: CoupledChargeLeafContext,
    *,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    charge_C: float,
    time_s: float,
) -> _StageDynamics:
    particle_mass = float(context.motion.particle_mass_kg)
    q_over_m = float(charge_C) / particle_mass
    return _sample_stage_dynamics(
        replace(context.motion, electric_q_over_m_Ckg=float(q_over_m)),
        position_m=position_m,
        velocity_mps=velocity_mps,
        time_s=float(time_s),
    )


def _boole_ballistic_endpoint(
    *,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    start: _StageDynamics,
    quarter: _StageDynamics,
    midpoint: _StageDynamics,
    three_quarter: _StageDynamics,
    endpoint: _StageDynamics,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    duration = float(duration_s)
    acceleration_start = np.asarray(start.body_acceleration_mps2, dtype=np.float64)
    acceleration_quarter = np.asarray(quarter.body_acceleration_mps2, dtype=np.float64)
    acceleration_mid = np.asarray(midpoint.body_acceleration_mps2, dtype=np.float64)
    acceleration_three_quarter = np.asarray(
        three_quarter.body_acceleration_mps2, dtype=np.float64
    )
    acceleration_end = np.asarray(endpoint.body_acceleration_mps2, dtype=np.float64)
    velocity = (
        np.asarray(velocity_mps, dtype=np.float64)
        + duration
        * (
            7.0 * acceleration_start
            + 32.0 * acceleration_quarter
            + 12.0 * acceleration_mid
            + 32.0 * acceleration_three_quarter
            + 7.0 * acceleration_end
        )
        / 90.0
    )
    position = (
        np.asarray(position_m, dtype=np.float64)
        + duration * np.asarray(velocity_mps, dtype=np.float64)
        + duration
        * duration
        * (
            7.0 * acceleration_start
            + 24.0 * acceleration_quarter
            + 6.0 * acceleration_mid
            + 8.0 * acceleration_three_quarter
        )
        / 90.0
    )
    return position, velocity


def advance_coupled_charge_leaf(
    context: CoupledChargeLeafContext,
    *,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    charge_C: float,
    start_time_s: float,
    duration_s: float,
) -> CoupledChargeLeafState:
    duration = max(float(duration_s), 0.0)
    start_position = np.asarray(position_m, dtype=np.float64)
    start_velocity = np.asarray(velocity_mps, dtype=np.float64)
    if duration == 0.0:
        return CoupledChargeLeafState(
            position_m=start_position.copy(),
            velocity_mps=start_velocity.copy(),
            charge_C=float(charge_C),
            sample_positions_m=np.empty((0, start_position.size), dtype=np.float64),
        )

    half_duration = 0.5 * duration
    midpoint_time = float(start_time_s) + half_duration
    start_dynamics = _sample_motion(
        context,
        position_m=start_position,
        velocity_mps=start_velocity,
        charge_C=float(charge_C),
        time_s=float(start_time_s),
    )
    coefficient_position, coefficient_velocity = _advance_with_stage_dynamics(
        context.motion,
        position_m=start_position,
        velocity_mps=start_velocity,
        dynamics=start_dynamics,
        duration_s=half_duration,
    )
    midpoint_charge = _advance_charge(
        context,
        charge_C=float(charge_C),
        position_m=start_position,
        time_s=float(start_time_s),
        duration_s=half_duration,
    )
    midpoint_dynamics = _sample_motion(
        context,
        position_m=coefficient_position,
        velocity_mps=coefficient_velocity,
        charge_C=midpoint_charge,
        time_s=midpoint_time,
    )
    half_position, _half_velocity = _advance_with_affine_stage_dynamics(
        position_m=start_position,
        velocity_mps=start_velocity,
        start=start_dynamics,
        midpoint=midpoint_dynamics,
        stage_fraction=0.5,
        duration_s=half_duration,
    )
    endpoint_position, endpoint_velocity = _advance_with_affine_stage_dynamics(
        position_m=start_position,
        velocity_mps=start_velocity,
        start=start_dynamics,
        midpoint=midpoint_dynamics,
        stage_fraction=1.0,
        duration_s=duration,
    )
    endpoint_charge = _advance_charge(
        context,
        charge_C=float(midpoint_charge),
        position_m=endpoint_position,
        time_s=float(start_time_s) + duration,
        duration_s=half_duration,
    )
    sample_positions = [
        coefficient_position,
        half_position,
        endpoint_position,
    ]
    if int(context.motion.drag_model_mode) == int(DRAG_MODEL_NONE):
        dense_half_stage_charge = _advance_charge(
            context,
            charge_C=float(charge_C),
            position_m=start_position,
            time_s=float(start_time_s),
            duration_s=0.25 * duration,
        )
        dense_half_charge = _advance_charge(
            context,
            charge_C=float(dense_half_stage_charge),
            position_m=half_position,
            time_s=midpoint_time,
            duration_s=0.25 * duration,
        )
        quarter_position, quarter_velocity = _advance_with_affine_stage_dynamics(
            position_m=start_position,
            velocity_mps=start_velocity,
            start=start_dynamics,
            midpoint=midpoint_dynamics,
            stage_fraction=0.25,
            duration_s=0.25 * duration,
        )
        three_quarter_position, three_quarter_velocity = (
            _advance_with_affine_stage_dynamics(
                position_m=start_position,
                velocity_mps=start_velocity,
                start=start_dynamics,
                midpoint=midpoint_dynamics,
                stage_fraction=0.75,
                duration_s=0.75 * duration,
            )
        )
        quarter_stage_charge = _advance_charge(
            context,
            charge_C=float(charge_C),
            position_m=start_position,
            time_s=float(start_time_s),
            duration_s=0.125 * duration,
        )
        quarter_charge = _advance_charge(
            context,
            charge_C=float(quarter_stage_charge),
            position_m=quarter_position,
            time_s=float(start_time_s) + 0.25 * duration,
            duration_s=0.125 * duration,
        )
        three_quarter_stage_charge = _advance_charge(
            context,
            charge_C=float(dense_half_charge),
            position_m=half_position,
            time_s=midpoint_time,
            duration_s=0.125 * duration,
        )
        three_quarter_charge = _advance_charge(
            context,
            charge_C=float(three_quarter_stage_charge),
            position_m=three_quarter_position,
            time_s=float(start_time_s) + 0.75 * duration,
            duration_s=0.125 * duration,
        )
        quarter_dynamics = _sample_motion(
            context,
            position_m=quarter_position,
            velocity_mps=quarter_velocity,
            charge_C=quarter_charge,
            time_s=float(start_time_s) + 0.25 * duration,
        )
        three_quarter_dynamics = _sample_motion(
            context,
            position_m=three_quarter_position,
            velocity_mps=three_quarter_velocity,
            charge_C=three_quarter_charge,
            time_s=float(start_time_s) + 0.75 * duration,
        )
        endpoint_dynamics = _sample_motion(
            context,
            position_m=endpoint_position,
            velocity_mps=endpoint_velocity,
            charge_C=endpoint_charge,
            time_s=float(start_time_s) + duration,
        )
        sample_positions.extend((quarter_position, three_quarter_position))
        endpoint_position, endpoint_velocity = _boole_ballistic_endpoint(
            position_m=start_position,
            velocity_mps=start_velocity,
            start=start_dynamics,
            quarter=quarter_dynamics,
            midpoint=midpoint_dynamics,
            three_quarter=three_quarter_dynamics,
            endpoint=endpoint_dynamics,
            duration_s=duration,
        )
        sample_positions.append(endpoint_position)
    return CoupledChargeLeafState(
        position_m=np.asarray(endpoint_position, dtype=np.float64),
        velocity_mps=np.asarray(endpoint_velocity, dtype=np.float64),
        charge_C=float(endpoint_charge),
        sample_positions_m=np.asarray(sample_positions, dtype=np.float64),
    )


def _coupled_error_requires_refinement(
    *,
    position_start_m: np.ndarray,
    velocity_start_mps: np.ndarray,
    charge_start_C: float,
    full: CoupledChargeLeafState,
    refined: CoupledChargeLeafState,
    duration_s: float,
) -> bool:
    for axis in range(position_start_m.size):
        if etd2_position_error_exceeds_tolerance(
            float(position_start_m[axis]),
            float(full.position_m[axis]),
            float(refined.position_m[axis]),
            float(velocity_start_mps[axis]),
            float(full.velocity_mps[axis]),
            float(refined.velocity_mps[axis]),
            float(duration_s),
        ) or etd2_velocity_error_exceeds_tolerance(
            float(velocity_start_mps[axis]),
            float(full.velocity_mps[axis]),
            float(refined.velocity_mps[axis]),
        ):
            return True
    return etd2_velocity_error_exceeds_tolerance(
        float(charge_start_C),
        float(full.charge_C),
        float(refined.charge_C),
    )


def advance_coupled_charge_embedded(
    context: CoupledChargeLeafContext,
    *,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    charge_C: float,
    start_time_s: float,
    duration_s: float,
) -> CoupledChargeEmbeddedStep:
    full = advance_coupled_charge_leaf(
        context,
        position_m=position_m,
        velocity_mps=velocity_mps,
        charge_C=float(charge_C),
        start_time_s=float(start_time_s),
        duration_s=float(duration_s),
    )
    half_duration = 0.5 * float(duration_s)
    refined_mid = advance_coupled_charge_leaf(
        context,
        position_m=position_m,
        velocity_mps=velocity_mps,
        charge_C=float(charge_C),
        start_time_s=float(start_time_s),
        duration_s=half_duration,
    )
    refined_end = advance_coupled_charge_leaf(
        context,
        position_m=refined_mid.position_m,
        velocity_mps=refined_mid.velocity_mps,
        charge_C=float(refined_mid.charge_C),
        start_time_s=float(start_time_s) + half_duration,
        duration_s=half_duration,
    )
    return CoupledChargeEmbeddedStep(
        full=full,
        refined_mid=refined_mid,
        refined_end=refined_end,
        refinement_required=_coupled_error_requires_refinement(
            position_start_m=np.asarray(position_m, dtype=np.float64),
            velocity_start_mps=np.asarray(velocity_mps, dtype=np.float64),
            charge_start_C=float(charge_C),
            full=full,
            refined=refined_end,
            duration_s=float(duration_s),
        ),
    )


__all__ = (
    "CoupledChargeEmbeddedStep",
    "CoupledChargeLeafContext",
    "CoupledChargeLeafState",
    "advance_coupled_charge_embedded",
    "advance_coupled_charge_leaf",
    "coupled_charge_leaf_context",
)
