"""Scalar deterministic segment integration and valid-mask prefix replay."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from particle_tracer_unified.core.coordinate_systems import (
    canonicalize_axisymmetric_rz_positions,
)
from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    valid_mask_status_requires_stop,
)

from ._segment_motion_contracts import (
    SegmentMotionRequest,
    ValidMaskPrefixResolution,
)
from ._segment_stage_dynamics import _advance_etd2_substep
from .base_field_sampling import (
    sample_compiled_valid_mask_status as _sample_valid_mask_status,
)
from .integrator_common import (
    DRAG_MODEL_NONE,
    doubled_substep_count,
    etd2_position_error_exceeds_tolerance,
    etd2_velocity_error_exceeds_tolerance,
    uniform_substep_schedule,
)


@dataclass(frozen=True, slots=True)
class SegmentMotionTrace:
    """Accepted corrected motion nodes and coefficient-sampling locations.

    ``positions_m`` holds corrected half/end nodes used by collision queries;
    ``coefficient_midpoint_positions_m`` holds the predictor locations where
    the leaf coefficients and relaxation time were sampled.
    """

    request: SegmentMotionRequest
    times_s: np.ndarray
    positions_m: np.ndarray
    coefficient_midpoint_positions_m: np.ndarray
    velocities_mps: np.ndarray
    support_status: np.ndarray
    tau_start_s: np.ndarray
    tau_mid_s: np.ndarray
    substep_count: int
    endpoint_position_m: np.ndarray
    endpoint_velocity_mps: np.ndarray
    aggregate_support_status: int
    local_error_resolved: bool

    def prefix(self, elapsed_s: float) -> SegmentMotionTrace:
        """Re-evaluate a prefix through the same canonical primitive."""

        return trace_motion_segment(self.request.prefix(float(elapsed_s)))

    def state_at(self, elapsed_s: float) -> tuple[np.ndarray, np.ndarray]:
        elapsed = float(
            np.clip(float(elapsed_s), 0.0, max(float(self.request.duration_s), 0.0))
        )
        if elapsed <= 0.0:
            return (
                np.asarray(self.request.position_m, dtype=np.float64).copy(),
                np.asarray(self.request.velocity_mps, dtype=np.float64).copy(),
            )
        if elapsed >= float(self.request.duration_s):
            return self.endpoint_position_m.copy(), self.endpoint_velocity_mps.copy()

        duration = float(self.request.duration_s)
        leaf_count = max(1, int(self.substep_count))
        half_leaf = duration / float(2 * leaf_count)
        node_index = round(elapsed / half_leaf) - 1
        # Every saved row is evaluated on the accepted affine trajectory;
        # reuse exact half/end nodes before evaluating a partial leaf.
        if 0 <= node_index < self.times_s.size:
            node_elapsed = float(self.times_s[node_index]) - self.request.start_time_s
            if elapsed == node_elapsed:
                return (
                    self.positions_m[node_index].copy(),
                    self.velocities_mps[node_index].copy(),
                )

        leaf_duration = duration / float(leaf_count)
        leaf_index = min(int(elapsed / leaf_duration), leaf_count - 1)
        leaf_start_elapsed = float(leaf_index) * leaf_duration
        if leaf_index == 0:
            leaf_position = np.asarray(self.request.position_m, dtype=np.float64)
            leaf_velocity = np.asarray(self.request.velocity_mps, dtype=np.float64)
        else:
            saved_endpoint = 2 * leaf_index - 1
            leaf_position = self.positions_m[saved_endpoint]
            leaf_velocity = self.velocities_mps[saved_endpoint]
        partial_duration = elapsed - leaf_start_elapsed
        partial_position, partial_velocity, *_ = _advance_request_substep(
            self.request,
            position_m=leaf_position,
            velocity_mps=leaf_velocity,
            duration_s=partial_duration,
            start_time_s=self.request.start_time_s + leaf_start_elapsed,
            tau_stokes_s=float(self.request.tau_stokes_s),
            body_acceleration_mps2=np.asarray(
                self.request.body_acceleration_mps2,
                dtype=np.float64,
            )[: self.request.spatial_dim],
        )
        return partial_position, partial_velocity


def _stationary_trace(
    request: SegmentMotionRequest,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
) -> SegmentMotionTrace:
    return SegmentMotionTrace(
        request=request,
        times_s=np.asarray([float(request.end_time_s)], dtype=np.float64),
        positions_m=position_m.reshape(1, position_m.size).copy(),
        coefficient_midpoint_positions_m=np.empty(
            (0, position_m.size), dtype=np.float64
        ),
        velocities_mps=velocity_mps.reshape(1, velocity_mps.size).copy(),
        support_status=np.asarray([VALID_MASK_STATUS_CLEAN], dtype=np.uint8),
        tau_start_s=np.empty(0, dtype=np.float64),
        tau_mid_s=np.empty(0, dtype=np.float64),
        substep_count=1,
        endpoint_position_m=position_m,
        endpoint_velocity_mps=velocity_mps,
        aggregate_support_status=int(VALID_MASK_STATUS_CLEAN),
        local_error_resolved=True,
    )


def _advance_request_substep(
    request: SegmentMotionRequest,
    *,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    duration_s: float,
    start_time_s: float,
    tau_stokes_s: float,
    body_acceleration_mps2: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    np.ndarray,
]:
    return _advance_etd2_substep(
        x0=position_m,
        v0=velocity_mps,
        dt_sub=float(duration_s),
        t_sub_start=float(start_time_s),
        spatial_dim=int(request.spatial_dim),
        compiled=request.backend,
        body=body_acceleration_mps2,
        tau_stokes=float(tau_stokes_s),
        particle_diameter_m=float(request.particle_diameter_m),
        particle_density_kgm3=float(request.particle_density_kgm3),
        particle_mass_kg=float(request.particle_mass_kg),
        dep_particle_rel_permittivity=float(request.dep_particle_rel_permittivity),
        thermophoretic_coeff=float(request.thermophoretic_coefficient),
        gas_density_kgm3=float(request.gas_density_kgm3),
        gas_mu_pas=float(request.gas_dynamic_viscosity_Pas),
        gas_temperature_K=float(request.gas_temperature_K),
        gas_molecular_mass_kg=float(request.gas_molecular_mass_kg),
        drag_model_mode=int(request.drag_model_mode),
        electric_q_over_m_i=request.electric_q_over_m_Ckg,
        force_runtime=request.force_runtime,
    )


def _advance_request_embedded_substep(
    request: SegmentMotionRequest,
    *,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    duration_s: float,
    start_time_s: float,
    tau_stokes_s: float,
    body_acceleration_mps2: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    return _advance_etd2_substep(
        x0=position_m,
        v0=velocity_mps,
        dt_sub=float(duration_s),
        t_sub_start=float(start_time_s),
        spatial_dim=int(request.spatial_dim),
        compiled=request.backend,
        body=body_acceleration_mps2,
        tau_stokes=float(tau_stokes_s),
        particle_diameter_m=float(request.particle_diameter_m),
        particle_density_kgm3=float(request.particle_density_kgm3),
        particle_mass_kg=float(request.particle_mass_kg),
        dep_particle_rel_permittivity=float(request.dep_particle_rel_permittivity),
        thermophoretic_coeff=float(request.thermophoretic_coefficient),
        gas_density_kgm3=float(request.gas_density_kgm3),
        gas_mu_pas=float(request.gas_dynamic_viscosity_Pas),
        gas_temperature_K=float(request.gas_temperature_K),
        gas_molecular_mass_kg=float(request.gas_molecular_mass_kg),
        drag_model_mode=int(request.drag_model_mode),
        electric_q_over_m_i=request.electric_q_over_m_Ckg,
        force_runtime=request.force_runtime,
        estimate_local_error=True,
    )


def _local_error_requires_refinement(
    position_start_m: np.ndarray,
    velocity_start_mps: np.ndarray,
    position_full_m: np.ndarray,
    velocity_full_mps: np.ndarray,
    position_refined_m: np.ndarray,
    velocity_refined_mps: np.ndarray,
    duration_s: float,
) -> bool:
    for axis in range(position_start_m.size):
        if etd2_position_error_exceeds_tolerance(
            float(position_start_m[axis]),
            float(position_full_m[axis]),
            float(position_refined_m[axis]),
            float(velocity_start_mps[axis]),
            float(velocity_full_mps[axis]),
            float(velocity_refined_mps[axis]),
            float(duration_s),
        ) or etd2_velocity_error_exceeds_tolerance(
            float(velocity_start_mps[axis]),
            float(velocity_full_mps[axis]),
            float(velocity_refined_mps[axis]),
        ):
            return True
    return False


def _trace_fixed_schedule(
    request: SegmentMotionRequest,
    *,
    substep_count: int,
    substep_duration_s: float,
    start_time_s: float,
    tau_stokes_s: float,
    body_acceleration_mps2: np.ndarray,
    estimate_local_error: bool,
) -> tuple[SegmentMotionTrace, bool]:
    """Evaluate one uniform schedule without accepting or refining it."""

    spatial_dim = int(request.spatial_dim)
    backend = request.backend
    x_curr = np.asarray(request.position_m, dtype=np.float64)[:spatial_dim].copy()
    v_curr = np.asarray(request.velocity_mps, dtype=np.float64)[:spatial_dim].copy()
    stage_trace: list[np.ndarray] = []
    coefficient_midpoint_trace: list[np.ndarray] = []
    velocity_trace: list[np.ndarray] = []
    support_trace: list[int] = []
    tau_start_trace: list[float] = []
    tau_mid_trace: list[float] = []
    valid_mask_status = int(VALID_MASK_STATUS_CLEAN)
    refinement_required = False
    for sub_idx in range(substep_count):
        t_sub_start = start_time_s + float(sub_idx) * substep_duration_s
        x_prev = x_curr.copy()
        v_prev = v_curr.copy()
        sample_x_prev = (
            canonicalize_axisymmetric_rz_positions(x_prev)
            if str(getattr(backend, "coordinate_system", "")) == "axisymmetric_rz"
            else x_prev
        )
        valid_mask_status = max(
            valid_mask_status,
            int(_sample_valid_mask_status(backend, sample_x_prev)),
        )
        if bool(estimate_local_error) and not refinement_required:
            (
                x_curr,
                v_curr,
                x_half,
                v_half,
                tau_start,
                tau_mid,
                coefficient_midpoint,
                refined_mid_position,
                refined_mid_velocity,
                refined_position,
                refined_velocity,
            ) = _advance_request_embedded_substep(
                request,
                position_m=x_prev,
                velocity_mps=v_prev,
                duration_s=float(substep_duration_s),
                start_time_s=float(t_sub_start),
                tau_stokes_s=float(tau_stokes_s),
                body_acceleration_mps2=body_acceleration_mps2,
            )
            refinement_required = _local_error_requires_refinement(
                x_prev,
                v_prev,
                x_curr,
                v_curr,
                refined_position,
                refined_velocity,
                float(substep_duration_s),
            )
            x_half = refined_mid_position
            v_half = refined_mid_velocity
        else:
            (
                x_curr,
                v_curr,
                x_half,
                v_half,
                tau_start,
                tau_mid,
                coefficient_midpoint,
            ) = _advance_request_substep(
                request,
                position_m=x_prev,
                velocity_mps=v_prev,
                duration_s=float(substep_duration_s),
                start_time_s=float(t_sub_start),
                tau_stokes_s=float(tau_stokes_s),
                body_acceleration_mps2=body_acceleration_mps2,
            )
        if str(getattr(backend, "coordinate_system", "")) == "axisymmetric_rz":
            sample_x_half = canonicalize_axisymmetric_rz_positions(x_half)
            sample_x_curr = canonicalize_axisymmetric_rz_positions(x_curr)
            sample_coefficient_midpoint = canonicalize_axisymmetric_rz_positions(
                coefficient_midpoint
            )
        else:
            sample_x_half = x_half
            sample_x_curr = x_curr
            sample_coefficient_midpoint = coefficient_midpoint
        half_status = max(
            int(_sample_valid_mask_status(backend, sample_x_half)),
            int(_sample_valid_mask_status(backend, sample_coefficient_midpoint)),
        )
        endpoint_status = int(_sample_valid_mask_status(backend, sample_x_curr))
        valid_mask_status = max(valid_mask_status, half_status, endpoint_status)
        # The trace is uniformly timed: half/end for every accepted substep.
        stage_trace.extend((x_half.copy(), x_curr.copy()))
        coefficient_midpoint_trace.append(sample_coefficient_midpoint.copy())
        velocity_trace.extend((v_half.copy(), v_curr.copy()))
        support_trace.extend((half_status, endpoint_status))
        tau_start_trace.append(float(tau_start))
        tau_mid_trace.append(float(tau_mid))

    stage_times = start_time_s + np.arange(
        1,
        2 * substep_count + 1,
        dtype=np.float64,
    ) * (0.5 * substep_duration_s)
    return (
        SegmentMotionTrace(
            request=request,
            times_s=stage_times,
            positions_m=np.asarray(stage_trace, dtype=np.float64),
            coefficient_midpoint_positions_m=np.asarray(
                coefficient_midpoint_trace, dtype=np.float64
            ),
            velocities_mps=np.asarray(velocity_trace, dtype=np.float64),
            support_status=np.asarray(support_trace, dtype=np.uint8),
            tau_start_s=np.asarray(tau_start_trace, dtype=np.float64),
            tau_mid_s=np.asarray(tau_mid_trace, dtype=np.float64),
            substep_count=int(max(1, substep_count)),
            endpoint_position_m=x_curr,
            endpoint_velocity_mps=v_curr,
            aggregate_support_status=int(valid_mask_status),
            local_error_resolved=not bool(refinement_required),
        ),
        refinement_required,
    )


def trace_motion_segment(request: SegmentMotionRequest) -> SegmentMotionTrace:
    """Advance one segment and return every accepted ETD2 half/end stage."""

    spatial_dim = int(request.spatial_dim)
    dt_seg = float(max(request.duration_s, 0.0))
    position = np.asarray(request.position_m, dtype=np.float64)[:spatial_dim].copy()
    velocity = np.asarray(request.velocity_mps, dtype=np.float64)[:spatial_dim].copy()
    if dt_seg <= 0.0:
        return _stationary_trace(request, position, velocity)

    tau_stokes = float(request.tau_stokes_s)
    if int(request.drag_model_mode) != int(DRAG_MODEL_NONE) and (
        not np.isfinite(tau_stokes) or tau_stokes <= 0.0
    ):
        raise ValueError(
            "effective particle drag relaxation time must be finite and > 0"
        )
    t_start = request.start_time_s
    n_substeps, dt_sub, t_start = uniform_substep_schedule(
        dt_seg,
        request.end_time_s,
        int(request.adaptive_substep_max_splits),
        int(request.minimum_substeps),
    )
    body = np.asarray(request.body_acceleration_mps2, dtype=np.float64)[:spatial_dim]
    while True:
        refined_count = int(
            doubled_substep_count(
                int(n_substeps),
                int(request.adaptive_substep_max_splits),
            )
        )
        estimate_local_error = bool(int(request.adaptive_substep_enabled) != 0)
        trace, refinement_required = _trace_fixed_schedule(
            request,
            substep_count=int(n_substeps),
            substep_duration_s=float(dt_sub),
            start_time_s=float(t_start),
            tau_stokes_s=float(tau_stokes),
            body_acceleration_mps2=body,
            estimate_local_error=estimate_local_error,
        )
        if not refinement_required:
            return trace
        if refined_count == n_substeps:
            return trace
        n_substeps = refined_count
        dt_sub = dt_seg / float(n_substeps)


def resolve_valid_mask_prefix(
    request: SegmentMotionRequest,
    *,
    max_halving_count: int,
    require_clean_prefix: bool = False,
) -> ValidMaskPrefixResolution:
    x_start = np.asarray(request.position_m, dtype=np.float64).copy()
    v_start = np.asarray(request.velocity_mps, dtype=np.float64).copy()
    dt_seg = max(float(request.duration_s), 0.0)
    halving_limit = int(max(0, max_halving_count))
    if dt_seg <= 0.0 or halving_limit <= 0:
        return ValidMaskPrefixResolution(
            position=x_start,
            velocity=v_start,
            accepted_dt=0.0,
            retry_count=0,
            found_valid_prefix=False,
        )

    retry_count = 0
    for split_idx in range(1, halving_limit + 1):
        retry_count += 1
        prefix_dt = float(dt_seg) * (0.5 ** int(split_idx))
        trace = trace_motion_segment(request.prefix(prefix_dt))
        retry_status = trace.aggregate_support_status
        status_ok = (
            int(retry_status) == int(VALID_MASK_STATUS_CLEAN)
            if bool(require_clean_prefix)
            else not bool(valid_mask_status_requires_stop(int(retry_status)))
        )
        if bool(trace.local_error_resolved) and bool(status_ok):
            return ValidMaskPrefixResolution(
                position=trace.endpoint_position_m,
                velocity=trace.endpoint_velocity_mps,
                accepted_dt=float(prefix_dt),
                retry_count=int(retry_count),
                found_valid_prefix=True,
            )

    return ValidMaskPrefixResolution(
        position=x_start,
        velocity=v_start,
        accepted_dt=0.0,
        retry_count=int(retry_count),
        found_valid_prefix=False,
    )
