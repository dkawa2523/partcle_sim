"""Resolve Brownian OU coefficients on a physics-accurate leaf schedule."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ._stochastic_composition import trace_particle_motion
from ._stochastic_config import StochasticMotionConfig
from ._stochastic_path import _integrated_ou_covariances
from ._stochastic_temperature import ParticleLeafPlan, sample_plan_temperatures
from .field_runtime import measure_sample_fields_for_stage, sample_fields_for_stage
from .integrator_common import (
    DRAG_MODEL_STOKES,
    ETD2_LOCAL_ERROR_RTOL,
    ETD2_STEP_DOUBLING_COARSE_ERROR_FACTOR,
)
from .sampling_backend import GAS_DENSITY
from .segment_motion import SegmentMotionBatchTrace, trace_motion_segment

K_BOLTZMANN = 1.380649e-23


@dataclass(frozen=True, slots=True)
class SampledLeafPlan:
    plan: ParticleLeafPlan
    temperatures_K: np.ndarray
    effective_masses_kg: np.ndarray

    @property
    def thermal_velocity_variance_m2s2(self) -> np.ndarray:
        return (
            K_BOLTZMANN
            * np.asarray(self.temperatures_K, dtype=np.float64)
            / np.asarray(self.effective_masses_kg, dtype=np.float64)
        )


def _collect_leaf_plan(
    motion_batch: SegmentMotionBatchTrace,
    particle_index: int,
    substep_count: int,
    particle_mass: np.ndarray,
) -> tuple[ParticleLeafPlan, bool]:
    index = int(particle_index)
    mass = float(np.asarray(particle_mass, dtype=np.float64)[index])
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(
            f"Brownian motion requires finite positive mass_kg; particle index {index}"
        )
    counts = np.asarray(motion_batch.substep_count, dtype=np.int32).copy()
    counts[index] = int(max(1, substep_count))
    requested_count = int(max(1, substep_count))
    configured_splits = int(max(0, motion_batch.request.adaptive_substep_max_splits))
    required_splits = int(max(0, requested_count - 1).bit_length())
    if required_splits <= configured_splits:
        trace = trace_particle_motion(motion_batch, index, counts)
    else:
        request = motion_batch.request.particle_request(index)
        request = request.with_minimum_substeps(requested_count)
        trace = trace_motion_segment(
            replace(
                request,
                adaptive_substep_max_splits=required_splits,
            )
        )
    leaf_count = int(trace.substep_count)
    tau_start = np.asarray(trace.tau_start_s, dtype=np.float64)
    tau_mid = np.asarray(trace.tau_mid_s, dtype=np.float64)
    if tau_start.shape != (leaf_count,) or tau_mid.shape != (leaf_count,):
        raise RuntimeError(
            "motion trace did not provide one drag coefficient per accepted leaf"
        )
    if np.any(~np.isfinite(tau_start) | (tau_start <= 0.0)) or np.any(
        ~np.isfinite(tau_mid) | (tau_mid <= 0.0)
    ):
        raise ValueError(
            "Brownian motion received invalid accepted-leaf drag tau; "
            f"particle index {index}"
        )
    return (
        ParticleLeafPlan(
            particle_index=index,
            leaf_end_times_s=np.asarray(
                trace.times_s[1::2] - trace.request.start_time_s, dtype=np.float64
            ),
            midpoint_times_s=np.asarray(trace.times_s[0::2], dtype=np.float64),
            midpoint_positions_m=np.asarray(
                trace.coefficient_midpoint_positions_m, dtype=np.float64
            ),
            tau_mid_s=tau_mid,
            particle_mass_kg=mass,
        ),
        bool(trace.local_error_resolved),
    )


def _sample_midpoint_gas_density(
    *,
    motion_batch: SegmentMotionBatchTrace,
    plan: ParticleLeafPlan,
    collect_diagnostics: bool,
) -> tuple[np.ndarray, float, int, int]:
    request = motion_batch.request
    output = np.empty(plan.leaf_end_times_s.size, dtype=np.float64)
    elapsed_total = 0.0
    point_total = 0
    call_total = 0
    for leaf_row, (time_s, point) in enumerate(
        zip(plan.midpoint_times_s, plan.midpoint_positions_m, strict=True)
    ):
        args = (
            request.backend,
            None,
            np.asarray([point], dtype=np.float64),
            float(time_s),
        )
        kwargs = {
            "spatial_dim": int(request.spatial_dim),
            "need_gas_density": True,
            "need_valid_mask": False,
            "fallback_density_kgm3": float(request.gas_density_kgm3),
        }
        if collect_diagnostics:
            sampled, metrics = measure_sample_fields_for_stage(*args, **kwargs)
            elapsed_total += float(metrics.elapsed_s)
            point_total += int(metrics.point_count)
            call_total += int(metrics.call_count)
        else:
            sampled = sample_fields_for_stage(*args, **kwargs)
        output[leaf_row] = float(sampled.values[GAS_DENSITY][0])
    return output, elapsed_total, point_total, call_total


def _effective_leaf_masses(
    *,
    motion_batch: SegmentMotionBatchTrace,
    plan: ParticleLeafPlan,
    collect_diagnostics: bool,
) -> tuple[np.ndarray, float, int, int]:
    runtime = motion_batch.request.force_runtime
    if runtime is None or not bool(runtime.virtual_mass_enabled):
        return (
            np.full(
                plan.leaf_end_times_s.size,
                float(plan.particle_mass_kg),
                dtype=np.float64,
            ),
            0.0,
            0,
            0,
        )
    density, elapsed, point_count, call_count = _sample_midpoint_gas_density(
        motion_batch=motion_batch,
        plan=plan,
        collect_diagnostics=collect_diagnostics,
    )
    rho_p = float(
        np.asarray(motion_batch.request.particle_density_kgm3, dtype=np.float64)[
            int(plan.particle_index)
        ]
    )
    factor = 1.0 + float(runtime.virtual_mass_coefficient) * density / rho_p
    return (
        float(plan.particle_mass_kg) * factor,
        elapsed,
        point_count,
        call_count,
    )


def _sample_plan(
    *,
    config: StochasticMotionConfig,
    motion_batch: SegmentMotionBatchTrace,
    plan: ParticleLeafPlan,
    gas_temperature_K: float,
    collect_diagnostics: bool,
) -> tuple[SampledLeafPlan, float, int, int]:
    temperatures, temperature_s, temperature_points, temperature_calls = (
        sample_plan_temperatures(
            config=config,
            compiled=motion_batch.request.backend,
            plans=[plan],
            spatial_dim=int(motion_batch.request.spatial_dim),
            gas_temperature_K=float(gas_temperature_K),
            collect_diagnostics=bool(collect_diagnostics),
        )
    )
    masses, density_s, density_points, density_calls = _effective_leaf_masses(
        motion_batch=motion_batch,
        plan=plan,
        collect_diagnostics=bool(collect_diagnostics),
    )
    return (
        SampledLeafPlan(
            plan=plan,
            temperatures_K=np.asarray(temperatures[0], dtype=np.float64),
            effective_masses_kg=np.asarray(masses, dtype=np.float64),
        ),
        float(temperature_s + density_s),
        int(temperature_points + density_points),
        int(temperature_calls + density_calls),
    )


def _ou_schedule_moments(sampled: SampledLeafPlan) -> tuple[np.ndarray, np.ndarray]:
    ends = np.asarray(sampled.plan.leaf_end_times_s, dtype=np.float64)
    starts = np.concatenate((np.asarray([0.0]), ends[:-1]))
    transition_total = np.eye(2, dtype=np.float64)
    covariance_total = np.zeros((2, 2), dtype=np.float64)
    for duration, tau, thermal in zip(
        ends - starts,
        sampled.plan.tau_mid_s,
        sampled.thermal_velocity_variance_m2s2,
        strict=True,
    ):
        one_minus_decay = -np.expm1(-float(duration) / float(tau))
        transition = np.asarray(
            (
                (1.0, float(tau) * one_minus_decay),
                (0.0, 1.0 - one_minus_decay),
            ),
            dtype=np.float64,
        )
        var_x, var_v, cov_xv = _integrated_ou_covariances(
            float(duration), float(tau), float(thermal)
        )
        covariance = np.asarray(((var_x, cov_xv), (cov_xv, var_v)))
        covariance_total = transition @ covariance_total @ transition.T + covariance
        transition_total = transition @ transition_total
    return transition_total, covariance_total


def _schedule_error(coarse: SampledLeafPlan, refined: SampledLeafPlan) -> float:
    duration = float(coarse.plan.leaf_end_times_s[-1])
    if duration <= 0.0 or duration != float(refined.plan.leaf_end_times_s[-1]):
        raise RuntimeError("Brownian coefficient schedules must span one interval")
    coarse_transition, coarse_covariance = _ou_schedule_moments(coarse)
    refined_transition, refined_covariance = _ou_schedule_moments(refined)
    tiny = float(np.finfo(np.float64).tiny)
    transition_scale = np.maximum(np.abs(refined_transition), tiny)
    covariance_scale = np.maximum(np.abs(refined_covariance), tiny)
    transition_defect = (
        np.abs(coarse_transition - refined_transition) / transition_scale
    )
    covariance_defect = (
        np.abs(coarse_covariance - refined_covariance) / covariance_scale
    )
    defect = max(
        float(np.max(transition_defect)),
        float(np.max(covariance_defect)),
    )
    return float(ETD2_STEP_DOUBLING_COARSE_ERROR_FACTOR) * defect


def _constant_by_construction(
    config: StochasticMotionConfig,
    motion_batch: SegmentMotionBatchTrace,
) -> bool:
    request = motion_batch.request
    runtime = request.force_runtime
    return bool(
        int(request.drag_model_mode) == int(DRAG_MODEL_STOKES)
        and (runtime is None or not bool(runtime.virtual_mass_enabled))
        and not str(request.backend.gas_mu_source).startswith("field:")
        and not (
            str(config.temperature_source) == "field_T_then_gas"
            and str(request.backend.gas_temperature_source).startswith("field:")
        )
    )


def _resolve_particle(
    *,
    config: StochasticMotionConfig,
    motion_batch: SegmentMotionBatchTrace,
    particle_index: int,
    initial_substeps: int,
    particle_mass: np.ndarray,
    gas_temperature_K: float,
    collect_diagnostics: bool,
) -> tuple[SampledLeafPlan, int, bool, float, int, int]:
    max_substeps = 1 << int(max(0, motion_batch.request.adaptive_substep_max_splits))
    plan, motion_resolved = _collect_leaf_plan(
        motion_batch, particle_index, initial_substeps, particle_mass
    )
    current, elapsed, points, calls = _sample_plan(
        config=config,
        motion_batch=motion_batch,
        plan=plan,
        gas_temperature_K=gas_temperature_K,
        collect_diagnostics=collect_diagnostics,
    )
    current_count = int(plan.leaf_end_times_s.size)
    if not motion_resolved:
        return current, current_count, False, elapsed, points, calls
    if _constant_by_construction(config, motion_batch):
        return current, current_count, True, elapsed, points, calls
    while True:
        at_limit = current_count >= max_substeps
        requested_count = (
            2 * current_count if at_limit else min(max_substeps, 2 * current_count)
        )
        candidate_plan, candidate_motion_resolved = _collect_leaf_plan(
            motion_batch, particle_index, requested_count, particle_mass
        )
        if not candidate_motion_resolved:
            return current, current_count, False, elapsed, points, calls
        candidate, extra_s, extra_points, extra_calls = _sample_plan(
            config=config,
            motion_batch=motion_batch,
            plan=candidate_plan,
            gas_temperature_K=gas_temperature_K,
            collect_diagnostics=collect_diagnostics,
        )
        elapsed += extra_s
        points += extra_points
        calls += extra_calls
        if _schedule_error(current, candidate) <= float(ETD2_LOCAL_ERROR_RTOL):
            return current, current_count, True, elapsed, points, calls
        if at_limit:
            return current, current_count, False, elapsed, points, calls
        current = candidate
        current_count = int(candidate_plan.leaf_end_times_s.size)


def resolve_coefficient_plans(
    *,
    config: StochasticMotionConfig,
    motion_batch: SegmentMotionBatchTrace,
    particle_indices: np.ndarray,
    minimum_substeps: np.ndarray,
    particle_mass: np.ndarray,
    gas_temperature_K: float,
    collect_diagnostics: bool,
) -> tuple[list[SampledLeafPlan], float, int, int]:
    counts = np.asarray(minimum_substeps)
    sampled_plans: list[SampledLeafPlan] = []
    elapsed_total = 0.0
    point_total = 0
    call_total = 0
    for raw_index in np.asarray(particle_indices, dtype=np.int64):
        index = int(raw_index)
        sampled, count, resolved, elapsed, points, calls = _resolve_particle(
            config=config,
            motion_batch=motion_batch,
            particle_index=index,
            initial_substeps=int(max(1, counts[index])),
            particle_mass=particle_mass,
            gas_temperature_K=float(gas_temperature_K),
            collect_diagnostics=bool(collect_diagnostics),
        )
        counts[index] = int(count)
        motion_batch.local_error_resolved[index] &= bool(resolved)
        if resolved:
            sampled_plans.append(sampled)
        elapsed_total += elapsed
        point_total += points
        call_total += calls
    return sampled_plans, elapsed_total, point_total, call_total
