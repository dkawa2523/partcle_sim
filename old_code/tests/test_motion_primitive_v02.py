from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from field_backend_helpers import (
    geometry_provider,
    regular_field_provider,
    write_triangle_mesh_field,
)

from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    QuantitySeriesND,
    RegularFieldND,
)
from particle_tracer_unified.providers.precomputed import (
    build_precomputed_triangle_mesh_field,
)
from particle_tracer_unified.solvers.field_compilation import compile_runtime_backend
from particle_tracer_unified.solvers.integrator_common import (
    DRAG_MODEL_EPSTEIN,
    DRAG_MODEL_NONE,
    DRAG_MODEL_SCHILLER_NAUMANN,
    DRAG_MODEL_STOKES,
    DRAG_MODEL_STOKES_CUNNINGHAM,
    stokes_relaxation_time,
)
from particle_tracer_unified.solvers.segment_motion import (
    SegmentMotionBatchRequest,
    trace_motion_batch,
    trace_motion_segment,
)


def _regular_backend(spatial_dim: int, *, include_viscosity: bool = True):
    axes = tuple(
        np.asarray([0.0, 1.0, 2.0], dtype=np.float64) for _ in range(spatial_dim)
    )
    shape = tuple(3 for _ in range(spatial_dim))
    valid = np.ones(shape, dtype=bool)
    quantities = {
        "ux": np.full(shape, 0.02, dtype=np.float64),
        "uy": np.full(shape, -0.01, dtype=np.float64),
        "rho_g": np.full(shape, 1.2, dtype=np.float64),
        "mu": np.full(shape, 1.8e-5, dtype=np.float64),
        "T": np.full(shape, 300.0, dtype=np.float64),
    }
    if not include_viscosity:
        del quantities["mu"]
    normals = [np.zeros(shape, dtype=np.float64) for _ in range(spatial_dim)]
    normals[-1].fill(1.0)
    field = regular_field_provider(axes, valid, quantities)
    geometry = geometry_provider(
        axes,
        valid,
        sdf=-np.ones(shape, dtype=np.float64),
        normal_components=tuple(normals),
    )
    return compile_runtime_backend(
        SimpleNamespace(
            geometry_provider=geometry,
            field_provider=field,
            gas=SimpleNamespace(
                density_kgm3=1.2,
                dynamic_viscosity_Pas=1.8e-5,
                temperature=300.0,
            ),
        ),
        spatial_dim=spatial_dim,
    )


def _viscosity_field_backend(
    viscosity: np.ndarray,
    times: np.ndarray,
):
    axes = (
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )
    valid = np.ones((3, 3), dtype=bool)
    time_values = np.asarray(times, dtype=np.float64)
    values = np.asarray(viscosity, dtype=np.float64)
    if values.shape != (time_values.size, 3, 3):
        raise ValueError("viscosity fixture must have shape (time, 3, 3)")
    zeros = np.zeros_like(values)
    quantities = {
        name: QuantitySeriesND(
            name=name,
            unit=unit,
            times=time_values,
            data=data,
        )
        for name, unit, data in (
            ("ux", "m/s", zeros),
            ("uy", "m/s", zeros),
            ("rho_g", "kg/m^3", np.full_like(values, 1.2)),
            ("mu", "Pa*s", values),
            ("T", "K", np.full_like(values, 300.0)),
        )
    }
    field = FieldProviderND(
        field=RegularFieldND(
            spatial_dim=2,
            coordinate_system="cartesian_xy",
            axis_names=("x", "y"),
            axes=axes,
            quantities=quantities,
            valid_mask=valid,
            time_mode="transient" if time_values.size > 1 else "steady",
        ),
        kind="synthetic",
    )
    normals = (np.zeros((3, 3)), np.ones((3, 3)))
    return compile_runtime_backend(
        SimpleNamespace(
            geometry_provider=geometry_provider(
                axes,
                valid,
                sdf=-np.ones((3, 3)),
                normal_components=normals,
            ),
            field_provider=field,
            gas=SimpleNamespace(
                density_kgm3=1.2,
                dynamic_viscosity_Pas=1.8e-5,
                temperature=300.0,
            ),
        ),
        spatial_dim=2,
    )


def _request(backend, spatial_dim: int, drag_model_mode: int, *, count: int = 1):
    mass = 1.0e-15
    diameter = 1.0e-6
    tau = (
        np.inf
        if int(drag_model_mode) == int(DRAG_MODEL_NONE)
        else stokes_relaxation_time(mass, 1.8e-5, diameter)
    )
    position = np.full((count, spatial_dim), 0.5, dtype=np.float64)
    velocity = np.zeros_like(position)
    velocity[:, 0] = 0.01
    return SegmentMotionBatchRequest(
        position_m=position,
        velocity_mps=velocity,
        active=np.ones(count, dtype=bool),
        tau_stokes_s=np.full(count, tau, dtype=np.float64),
        particle_diameter_m=np.full(count, diameter, dtype=np.float64),
        particle_density_kgm3=np.full(count, 1200.0, dtype=np.float64),
        particle_mass_kg=np.full(count, mass, dtype=np.float64),
        dep_particle_rel_permittivity=np.full(count, np.nan, dtype=np.float64),
        thermophoretic_coefficient=np.full(count, np.nan, dtype=np.float64),
        end_time_s=2.0e-4,
        duration_s=2.0e-4,
        spatial_dim=spatial_dim,
        backend=backend,
        body_acceleration_mps2=np.zeros(spatial_dim, dtype=np.float64),
        gas_density_kgm3=1.2,
        gas_dynamic_viscosity_Pas=1.8e-5,
        gas_temperature_K=300.0,
        gas_molecular_mass_kg=39.948 * 1.66053906660e-27,
        drag_model_mode=int(drag_model_mode),
        adaptive_substep_enabled=1,
        adaptive_substep_max_splits=4,
    )


@pytest.mark.parametrize("spatial_dim", [2, 3])
@pytest.mark.parametrize(
    "drag_model_mode",
    [
        DRAG_MODEL_NONE,
        DRAG_MODEL_STOKES,
        DRAG_MODEL_STOKES_CUNNINGHAM,
        DRAG_MODEL_SCHILLER_NAUMANN,
        DRAG_MODEL_EPSTEIN,
    ],
)
def test_regular_batch_and_candidate_replay_share_one_motion_rule(
    spatial_dim: int,
    drag_model_mode: int,
) -> None:
    request = _request(_regular_backend(spatial_dim), spatial_dim, drag_model_mode)

    batch = trace_motion_batch(request)
    replay = trace_motion_segment(request.particle_request(0))

    np.testing.assert_allclose(
        batch.endpoint_position_m[0],
        replay.endpoint_position_m,
        rtol=2.0e-12,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        batch.endpoint_velocity_mps[0],
        replay.endpoint_velocity_mps,
        rtol=2.0e-12,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        batch.midpoint_position_m[0],
        replay.positions_m[replay.substep_count - 1],
        rtol=2.0e-12,
        atol=2.0e-15,
    )
    assert int(batch.substep_count[0]) == replay.substep_count
    assert int(batch.aggregate_support_status[0]) == replay.aggregate_support_status
    assert batch.particle_trace(0).times_s.tolist() == pytest.approx(
        replay.times_s.tolist()
    )


def test_constant_viscosity_keeps_stokes_motion_bit_exact() -> None:
    field_request = _request(_regular_backend(2), 2, DRAG_MODEL_STOKES)
    fallback_request = _request(
        _regular_backend(2, include_viscosity=False),
        2,
        DRAG_MODEL_STOKES,
    )

    field_trace = trace_motion_batch(field_request)
    fallback_trace = trace_motion_batch(fallback_request)

    np.testing.assert_array_equal(
        field_trace.endpoint_position_m,
        fallback_trace.endpoint_position_m,
    )
    np.testing.assert_array_equal(
        field_trace.endpoint_velocity_mps,
        fallback_trace.endpoint_velocity_mps,
    )


def test_spatial_viscosity_field_sets_local_stokes_relaxation_in_both_paths() -> None:
    reference_viscosity = 1.8e-5
    x_axis = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    spatial_viscosity = reference_viscosity * (1.0 + x_axis[:, None])
    viscosity = np.broadcast_to(spatial_viscosity, (1, 3, 3)).copy()
    backend = _viscosity_field_backend(viscosity, np.asarray([0.0]))
    duration = 2.0e-5
    initial_speed = 0.1
    request = replace(
        _request(backend, 2, DRAG_MODEL_STOKES),
        position_m=np.asarray([[0.5, 0.25]], dtype=np.float64),
        velocity_mps=np.asarray([[0.0, initial_speed]], dtype=np.float64),
        end_time_s=duration,
        duration_s=duration,
        adaptive_substep_enabled=0,
        adaptive_substep_max_splits=0,
    )
    local_viscosity = 1.5 * reference_viscosity
    local_tau = float(request.particle_mass_kg[0]) / (
        3.0 * np.pi * local_viscosity * float(request.particle_diameter_m[0])
    )
    decay = np.exp(-duration / local_tau)
    expected_velocity = initial_speed * decay
    expected_y = 0.25 + local_tau * initial_speed * (1.0 - decay)

    batch = trace_motion_batch(request)
    scalar = trace_motion_segment(request.particle_request(0))

    assert batch.endpoint_velocity_mps[0, 1] == pytest.approx(
        expected_velocity,
        rel=2.0e-14,
    )
    assert scalar.endpoint_velocity_mps[1] == pytest.approx(
        expected_velocity,
        rel=2.0e-14,
    )
    assert batch.endpoint_position_m[0, 1] == pytest.approx(expected_y, rel=2.0e-14)
    assert scalar.endpoint_position_m[1] == pytest.approx(expected_y, rel=2.0e-14)


def test_transient_viscosity_field_matches_affine_stokes_decay_in_both_paths() -> None:
    duration = 2.0e-5
    viscosity_start = 1.8e-5
    viscosity_end = 5.4e-5
    viscosity = np.stack(
        [
            np.full((3, 3), viscosity_start, dtype=np.float64),
            np.full((3, 3), viscosity_end, dtype=np.float64),
        ]
    )
    backend = _viscosity_field_backend(
        viscosity,
        np.asarray([0.0, duration], dtype=np.float64),
    )
    initial_speed = 0.1
    request = replace(
        _request(backend, 2, DRAG_MODEL_STOKES),
        position_m=np.asarray([[0.5, 0.25]], dtype=np.float64),
        velocity_mps=np.asarray([[0.0, initial_speed]], dtype=np.float64),
        end_time_s=duration,
        duration_s=duration,
        adaptive_substep_enabled=0,
        adaptive_substep_max_splits=0,
    )
    integrated_viscosity = 0.5 * (viscosity_start + viscosity_end) * duration
    decay = np.exp(
        -3.0
        * np.pi
        * float(request.particle_diameter_m[0])
        * integrated_viscosity
        / float(request.particle_mass_kg[0])
    )
    expected_velocity = initial_speed * decay

    batch = trace_motion_batch(request)
    scalar = trace_motion_segment(request.particle_request(0))

    assert batch.endpoint_velocity_mps[0, 1] == pytest.approx(
        expected_velocity,
        rel=2.0e-14,
    )
    assert scalar.endpoint_velocity_mps[1] == pytest.approx(
        expected_velocity,
        rel=2.0e-14,
    )


def test_triangle_batch_and_candidate_replay_share_one_motion_rule(
    tmp_path: Path,
) -> None:
    field_path = write_triangle_mesh_field(tmp_path / "triangle_field.npz")
    field = build_precomputed_triangle_mesh_field(
        {"npz_path": str(field_path)},
        spatial_dim=2,
        coordinate_system="cartesian_xy",
    )
    axes = (np.asarray([0.0, 0.5, 1.0]), np.asarray([0.0, 0.5, 1.0]))
    valid = np.ones((3, 3), dtype=bool)
    geometry = geometry_provider(
        axes,
        valid,
        sdf=-np.ones((3, 3)),
        normal_components=(np.zeros((3, 3)), np.ones((3, 3))),
    )
    backend = compile_runtime_backend(
        SimpleNamespace(geometry_provider=geometry, field_provider=field),
        spatial_dim=2,
    )
    request = _request(backend, 2, DRAG_MODEL_STOKES)

    batch = trace_motion_batch(request)
    replay = trace_motion_segment(request.particle_request(0))

    np.testing.assert_allclose(
        batch.endpoint_position_m[0], replay.endpoint_position_m, rtol=2e-12
    )
    np.testing.assert_allclose(
        batch.endpoint_velocity_mps[0], replay.endpoint_velocity_mps, rtol=2e-12
    )
    assert int(batch.aggregate_support_status[0]) == replay.aggregate_support_status


def test_batch_trace_retained_memory_is_linear_in_particle_count() -> None:
    count = 4096
    request = _request(_regular_backend(2), 2, DRAG_MODEL_STOKES, count=count)

    trace = trace_motion_batch(request)

    retained_arrays = (
        trace.endpoint_position_m,
        trace.endpoint_velocity_mps,
        trace.midpoint_position_m,
        trace.substep_count,
        trace.aggregate_support_status,
    )
    retained_bytes = sum(array.nbytes for array in retained_arrays)
    expected_bytes = count * (3 * 2 * 8 + 4 + 1)
    assert retained_bytes == expected_bytes
    assert not hasattr(trace, "positions_m")
    assert not hasattr(trace, "velocities_mps")
