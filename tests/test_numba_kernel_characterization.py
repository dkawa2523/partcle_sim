from __future__ import annotations

import inspect
from collections.abc import Callable
from itertools import product

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numba import njit
from test_sampling_backend_v02 import _regular_backend, _triangle_backend

from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
)
from particle_tracer_unified.solvers import kernel2d_numba as grid2d_kernel
from particle_tracer_unified.solvers import (
    kernel2d_triangle_mesh_numba as triangle_kernel,
)
from particle_tracer_unified.solvers import kernel3d_numba as grid3d_kernel
from particle_tracer_unified.solvers import kernel_shared_numba as mask_kernel
from particle_tracer_unified.solvers import motion_kernel_numba as motion_kernel
from particle_tracer_unified.solvers.integrator_common import (
    _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
    DRAG_MODEL_SCHILLER_NAUMANN,
    DRAG_MODEL_STOKES,
    advance_state_2d,
    advance_state_3d,
)
from particle_tracer_unified.solvers.kernel_shared_numba import (
    mask_bilinear_status,
    mask_trilinear_status,
)
from particle_tracer_unified.solvers.motion_kernel_numba import (
    advance_etd2_batch_inplace,
)

_PARTICLE_STAGE = (0.2, 1.0e-6, 1200.0, 1.0e-15, 6.63e-26, DRAG_MODEL_STOKES)


@njit(cache=False)
def _constant_stage(
    particle_index,
    time_s,
    _px,
    _py,
    _pz,
    _vx,
    _vy,
    _vz,
    tau_stokes,
    _particle_diameter,
    _particle_density,
    _particle_mass,
    _gas_molecular_mass,
    _drag_model_mode,
    stage_values,
    _support_x_limit,
):
    status = VALID_MASK_STATUS_CLEAN
    if time_s >= stage_values[6]:
        status = VALID_MASK_STATUS_MIXED_STENCIL
    tau = tau_stokes
    if int(_drag_model_mode) != int(DRAG_MODEL_STOKES):
        tau = 0.5 * tau_stokes
    return (
        stage_values[0],
        stage_values[1],
        stage_values[2],
        stage_values[3],
        stage_values[4],
        stage_values[5],
        tau,
        status,
    )


@njit(cache=False)
def _x_bounded_support(x, _y, _z, _stage_values, support_x_limit):
    if x > support_x_limit:
        return VALID_MASK_STATUS_HARD_INVALID
    return VALID_MASK_STATUS_CLEAN


def _expected_constant_stage_motion(
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    dim: int,
    tau: float,
    stage_values: np.ndarray,
    substeps: int,
    duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_position = position.copy()
    current_velocity = velocity.copy()
    midpoint = np.full(dim, np.nan, dtype=np.float64)
    dt = duration / substeps
    for substep in range(substeps):
        if dim == 2:
            half_values = advance_state_2d(
                current_position[0],
                current_position[1],
                current_velocity[0],
                current_velocity[1],
                stage_values[0],
                stage_values[1],
                stage_values[3],
                stage_values[4],
                tau,
                0.5 * dt,
            )
            next_values = advance_state_2d(
                current_position[0],
                current_position[1],
                current_velocity[0],
                current_velocity[1],
                stage_values[0],
                stage_values[1],
                stage_values[3],
                stage_values[4],
                tau,
                dt,
            )
            half_position = np.asarray(half_values[:2])
            current_position = np.asarray(next_values[:2])
            current_velocity = np.asarray(next_values[2:])
        else:
            half_values = advance_state_3d(
                current_position[0],
                current_position[1],
                current_position[2],
                current_velocity[0],
                current_velocity[1],
                current_velocity[2],
                stage_values[0],
                stage_values[1],
                stage_values[2],
                stage_values[3],
                stage_values[4],
                stage_values[5],
                tau,
                0.5 * dt,
            )
            next_values = advance_state_3d(
                current_position[0],
                current_position[1],
                current_position[2],
                current_velocity[0],
                current_velocity[1],
                current_velocity[2],
                stage_values[0],
                stage_values[1],
                stage_values[2],
                stage_values[3],
                stage_values[4],
                stage_values[5],
                tau,
                dt,
            )
            half_position = np.asarray(half_values[:3])
            current_position = np.asarray(next_values[:3])
            current_velocity = np.asarray(next_values[3:])

        half_row = 2 * substep
        end_row = half_row + 1
        if half_row == substeps - 1:
            midpoint[:] = half_position
        if end_row == substeps - 1:
            midpoint[:] = current_position

    return current_position, current_velocity, midpoint


def test_mask_bilinear_status_classifies_all_corner_combinations() -> None:
    axis = np.asarray([0.0, 1.0], dtype=np.float64)

    for encoded_mask in range(2**4):
        mask = np.asarray(
            [(encoded_mask >> bit) & 1 for bit in range(4)],
            dtype=bool,
        ).reshape((2, 2))
        valid_corner_count = int(np.count_nonzero(mask))
        expected = VALID_MASK_STATUS_HARD_INVALID
        if valid_corner_count >= 2:
            expected = (
                VALID_MASK_STATUS_CLEAN
                if valid_corner_count == 4
                else VALID_MASK_STATUS_MIXED_STENCIL
            )

        status = mask_bilinear_status(mask, axis, axis, 0.5, 0.5)

        assert int(status) == expected

    assert getattr(mask_bilinear_status, "nopython_signatures", ())


def test_mask_bilinear_status_preserves_exact_boundaries_and_domain_limits() -> None:
    axis = np.asarray([0.0, 1.0], dtype=np.float64)
    for corner in product((0, 1), repeat=2):
        one_valid = np.zeros((2, 2), dtype=bool)
        one_valid[corner] = True
        assert (
            mask_bilinear_status(one_valid, axis, axis, *corner)
            == VALID_MASK_STATUS_MIXED_STENCIL
        )

        one_invalid = np.ones((2, 2), dtype=bool)
        one_invalid[corner] = False
        assert (
            mask_bilinear_status(one_invalid, axis, axis, *corner)
            == VALID_MASK_STATUS_HARD_INVALID
        )

    valid_mask = np.ones((2, 2), dtype=bool)
    epsilon = np.finfo(np.float64).eps
    outside_points = (
        (-epsilon, 0.5),
        (1.0 + epsilon, 0.5),
        (0.5, -epsilon),
        (0.5, 1.0 + epsilon),
    )
    for point in outside_points:
        assert (
            mask_bilinear_status(valid_mask, axis, axis, *point)
            == VALID_MASK_STATUS_HARD_INVALID
        )


def test_mask_bilinear_python_path_matches_compiled_classification(monkeypatch) -> None:
    axis = np.asarray([0.0, 1.0], dtype=np.float64)
    cases: list[tuple[np.ndarray, tuple[float, float]]] = []
    for encoded_mask in range(2**4):
        mask = np.asarray(
            [(encoded_mask >> bit) & 1 for bit in range(4)],
            dtype=bool,
        ).reshape((2, 2))
        cases.append((mask, (0.5, 0.5)))
    for corner in product((0, 1), repeat=2):
        point = (float(corner[0]), float(corner[1]))
        one_valid = np.zeros((2, 2), dtype=bool)
        one_valid[corner] = True
        cases.append((one_valid, point))
        one_invalid = np.ones((2, 2), dtype=bool)
        one_invalid[corner] = False
        cases.append((one_invalid, point))
    valid_mask = np.ones((2, 2), dtype=bool)
    cases.extend(
        [
            (valid_mask, (-1.0, 0.5)),
            (valid_mask, (0.5, 1.5)),
        ]
    )
    compiled_results = [
        mask_bilinear_status(mask, axis, axis, *point) for mask, point in cases
    ]
    python_mask_status = vars(mask_kernel.mask_bilinear_status)["py_func"]
    for helper_name in (
        "_outside_bilinear_domain",
        "_mask_weight",
        "_bilinear_mask_value",
        "_bilinear_corner_status",
    ):
        dispatcher = getattr(mask_kernel, helper_name)
        monkeypatch.setattr(mask_kernel, helper_name, dispatcher.py_func)

    python_results = [
        python_mask_status(mask, axis, axis, *point) for mask, point in cases
    ]

    assert python_results == compiled_results


def test_mask_bilinear_public_signature_is_stable() -> None:
    assert tuple(inspect.signature(mask_bilinear_status).parameters) == (
        "mask2d",
        "xs",
        "ys",
        "x",
        "y",
    )


def test_mask_trilinear_status_classifies_all_corner_combinations() -> None:
    axis = np.asarray([0.0, 1.0], dtype=np.float64)

    for encoded_mask in range(2**8):
        mask = np.asarray(
            [(encoded_mask >> bit) & 1 for bit in range(8)],
            dtype=bool,
        ).reshape((2, 2, 2))
        valid_corner_count = int(np.count_nonzero(mask))
        expected = VALID_MASK_STATUS_HARD_INVALID
        if valid_corner_count >= 4:
            expected = (
                VALID_MASK_STATUS_CLEAN
                if valid_corner_count == 8
                else VALID_MASK_STATUS_MIXED_STENCIL
            )

        status = mask_trilinear_status(
            mask,
            axis,
            axis,
            axis,
            0.5,
            0.5,
            0.5,
        )

        assert int(status) == expected

    assert getattr(mask_trilinear_status, "nopython_signatures", ())


def test_mask_trilinear_status_preserves_exact_boundaries_and_domain_limits() -> None:
    axis = np.asarray([0.0, 1.0], dtype=np.float64)
    for corner in product((0, 1), repeat=3):
        one_valid = np.zeros((2, 2, 2), dtype=bool)
        one_valid[corner] = True
        assert (
            mask_trilinear_status(one_valid, axis, axis, axis, *corner)
            == VALID_MASK_STATUS_MIXED_STENCIL
        )

        one_invalid = np.ones((2, 2, 2), dtype=bool)
        one_invalid[corner] = False
        assert (
            mask_trilinear_status(one_invalid, axis, axis, axis, *corner)
            == VALID_MASK_STATUS_HARD_INVALID
        )

    valid_mask = np.ones((2, 2, 2), dtype=bool)
    outside_points = (
        (-np.finfo(np.float64).eps, 0.5, 0.5),
        (1.0 + np.finfo(np.float64).eps, 0.5, 0.5),
        (0.5, -np.finfo(np.float64).eps, 0.5),
        (0.5, 1.0 + np.finfo(np.float64).eps, 0.5),
        (0.5, 0.5, -np.finfo(np.float64).eps),
        (0.5, 0.5, 1.0 + np.finfo(np.float64).eps),
    )
    for point in outside_points:
        assert (
            mask_trilinear_status(valid_mask, axis, axis, axis, *point)
            == VALID_MASK_STATUS_HARD_INVALID
        )


def test_mask_python_path_matches_compiled_classification(monkeypatch) -> None:
    axis = np.asarray([0.0, 1.0], dtype=np.float64)
    cases: list[tuple[np.ndarray, tuple[float, float, float]]] = []
    for encoded_mask in range(2**8):
        mask = np.asarray(
            [(encoded_mask >> bit) & 1 for bit in range(8)],
            dtype=bool,
        ).reshape((2, 2, 2))
        cases.append((mask, (0.5, 0.5, 0.5)))
    cases.extend(
        [
            (np.ones((2, 2, 2), dtype=bool), (-1.0, 0.5, 0.5)),
            (np.ones((2, 2, 2), dtype=bool), (0.5, 0.5, 1.5)),
        ]
    )
    compiled_results = [
        mask_trilinear_status(mask, axis, axis, axis, *point) for mask, point in cases
    ]
    python_mask_status = vars(mask_kernel.mask_trilinear_status)["py_func"]
    for helper_name in (
        "_outside_trilinear_domain",
        "_mask_weight",
        "_trilinear_mask_value",
        "_trilinear_corner_status",
    ):
        dispatcher = getattr(mask_kernel, helper_name)
        monkeypatch.setattr(mask_kernel, helper_name, dispatcher.py_func)

    python_results = [
        python_mask_status(mask, axis, axis, axis, *point) for mask, point in cases
    ]

    assert python_results == compiled_results


@pytest.mark.parametrize("dim", [2, 3])
def test_etd_batch_preserves_inactive_rows_and_commits_canonical_nodes(
    dim: int,
) -> None:
    position = np.asarray(
        [
            [-2.0, 0.25, 0.75],
            [0.0, -0.1, 0.2],
            [2.0, 0.5, -0.25],
        ],
        dtype=np.float64,
    )[:, :dim]
    velocity = np.asarray(
        [
            [0.4, -0.2, 0.1],
            [0.1, 0.2, -0.3],
            [-0.2, 0.3, 0.4],
        ],
        dtype=np.float64,
    )[:, :dim]
    original_position = position.copy()
    original_velocity = velocity.copy()
    active = np.asarray([False, True, True])
    tau = np.full(3, 0.2, dtype=np.float64)
    particle_diameter = np.full(3, 1.0e-6, dtype=np.float64)
    particle_density = np.full(3, 1200.0, dtype=np.float64)
    particle_mass = np.full(3, 1.0e-15, dtype=np.float64)
    stage_values = np.asarray(
        [0.3, -0.2, 0.1, 0.4, -0.3, 0.2, 0.15],
        dtype=np.float64,
    )
    endpoint_position = np.full_like(position, np.nan)
    endpoint_velocity = np.full_like(velocity, np.nan)
    midpoint_position = np.full_like(position, np.nan)
    substep_counts = np.zeros(3, dtype=np.int64)
    mask_statuses = np.full(3, -1, dtype=np.int64)
    local_error_resolved = np.zeros(3, dtype=bool)

    advance_etd2_batch_inplace(
        _constant_stage,
        _x_bounded_support,
        dim,
        position,
        velocity,
        active,
        tau,
        particle_diameter,
        particle_density,
        particle_mass,
        0.4,
        0.4,
        6.63e-26,
        DRAG_MODEL_STOKES,
        1,
        3,
        endpoint_position,
        endpoint_velocity,
        midpoint_position,
        substep_counts,
        mask_statuses,
        local_error_resolved,
        stage_values,
        1.0,
    )

    np.testing.assert_array_equal(position, original_position)
    np.testing.assert_array_equal(velocity, original_velocity)
    np.testing.assert_array_equal(endpoint_position[0], position[0])
    np.testing.assert_array_equal(endpoint_velocity[0], velocity[0])
    np.testing.assert_array_equal(midpoint_position[0], position[0])
    for particle_index in (1, 2):
        expected_position, expected_velocity, expected_midpoint = (
            _expected_constant_stage_motion(
                position[particle_index],
                velocity[particle_index],
                dim=dim,
                tau=tau[particle_index],
                stage_values=stage_values,
                substeps=1,
                duration=0.4,
            )
        )
        np.testing.assert_array_equal(
            endpoint_position[particle_index],
            expected_position,
        )
        np.testing.assert_array_equal(
            endpoint_velocity[particle_index],
            expected_velocity,
        )
        np.testing.assert_array_equal(
            midpoint_position[particle_index],
            expected_midpoint,
        )

    assert substep_counts.tolist() == [1, 1, 1]
    assert mask_statuses.tolist() == [
        VALID_MASK_STATUS_CLEAN,
        VALID_MASK_STATUS_MIXED_STENCIL,
        VALID_MASK_STATUS_HARD_INVALID,
    ]
    assert local_error_resolved.tolist() == [True, True, True]
    for output in (endpoint_position, endpoint_velocity, midpoint_position):
        assert output.shape == position.shape
        assert output.dtype == np.float64
        assert np.all(np.isfinite(output))
    assert getattr(advance_etd2_batch_inplace, "nopython_signatures", ())


def _motion_outputs(
    advancer: Callable[..., None],
    *,
    dim: int,
    drag_model_mode: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    position = np.asarray(
        [[-2.0, 0.25, 0.75], [0.0, -0.1, 0.2]],
        dtype=np.float64,
    )[:, :dim]
    velocity = np.asarray(
        [[0.4, -0.2, 0.1], [0.1, 0.2, -0.3]],
        dtype=np.float64,
    )[:, :dim]
    active = np.asarray([False, True])
    tau = np.full(2, 0.2, dtype=np.float64)
    particle_diameter = np.full(2, 1.0e-6, dtype=np.float64)
    particle_density = np.full(2, 1200.0, dtype=np.float64)
    particle_mass = np.full(2, 1.0e-15, dtype=np.float64)
    status_time = 0.0 if drag_model_mode == DRAG_MODEL_STOKES else 0.02
    support_x_limit = 1.0 if drag_model_mode == DRAG_MODEL_STOKES else -1.0
    stage_values = np.asarray(
        [0.3, -0.2, 0.1, 0.4, -0.3, 0.2, status_time],
        dtype=np.float64,
    )
    endpoint_position = np.full_like(position, np.nan)
    endpoint_velocity = np.full_like(velocity, np.nan)
    midpoint_position = np.full_like(position, np.nan)
    substep_counts = np.zeros(2, dtype=np.int64)
    mask_statuses = np.full(2, -1, dtype=np.int64)
    local_error_resolved = np.zeros(2, dtype=bool)
    advancer(
        _constant_stage,
        _x_bounded_support,
        dim,
        position,
        velocity,
        active,
        tau,
        particle_diameter,
        particle_density,
        particle_mass,
        0.3,
        0.3,
        6.63e-26,
        drag_model_mode,
        1,
        3,
        endpoint_position,
        endpoint_velocity,
        midpoint_position,
        substep_counts,
        mask_statuses,
        local_error_resolved,
        stage_values,
        support_x_limit,
    )
    return (
        endpoint_position,
        endpoint_velocity,
        midpoint_position,
        substep_counts,
        mask_statuses,
        local_error_resolved,
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "drag_model_mode",
    [DRAG_MODEL_STOKES, DRAG_MODEL_SCHILLER_NAUMANN],
)
def test_etd_python_path_matches_nopython_commits(
    monkeypatch,
    dim: int,
    drag_model_mode: int,
) -> None:
    compiled_outputs = _motion_outputs(
        advance_etd2_batch_inplace,
        dim=dim,
        drag_model_mode=drag_model_mode,
    )
    python_advancer = vars(motion_kernel.advance_etd2_batch_inplace)["py_func"]
    for helper_name in (
        "_commit_inactive_particle",
        "_larger_status",
        "_local_error_exceeds_tolerance",
        "_schedule_is_final",
        "_should_estimate_local_error",
        "_advance_spatial_state",
        "_advance_affine_spatial_state",
        "_advance_etd2_leaf",
        "_embedded_leaf_requires_refinement",
        "_write_position",
        "_capture_midpoint",
        "_commit_endpoint",
    ):
        dispatcher = getattr(motion_kernel, helper_name)
        monkeypatch.setattr(motion_kernel, helper_name, dispatcher.py_func)

    python_outputs = _motion_outputs(
        python_advancer,
        dim=dim,
        drag_model_mode=drag_model_mode,
    )

    for compiled, python in zip(compiled_outputs, python_outputs, strict=True):
        np.testing.assert_array_equal(python, compiled)


def test_etd_batch_public_signature_is_stable() -> None:
    assert tuple(inspect.signature(advance_etd2_batch_inplace).parameters) == (
        "stage_evaluator",
        "support_evaluator",
        "spatial_dim",
        "x",
        "v",
        "active",
        "tau_p",
        "particle_diameter",
        "particle_density",
        "particle_mass",
        "t_end",
        "duration",
        "gas_molecular_mass_kg",
        "drag_model_mode",
        "adaptive_substep_enabled",
        "adaptive_substep_max_splits",
        "x_end",
        "v_end",
        "x_mid",
        "substep_counts",
        "mask_status_flags",
        "local_error_resolved",
        "backend_payload",
    )


def _use_python_dispatchers(monkeypatch, *modules) -> None:
    for module in modules:
        for name, dispatcher in vars(module).items():
            python_function = getattr(dispatcher, "py_func", None)
            if python_function is not None:
                monkeypatch.setattr(module, name, python_function)


def _regular_leaf_arguments(backend, dim: int, invalid: bool):
    fills = (np.nan, 0.0, -1.0)
    thermodynamics = tuple(
        np.full_like(values, fills[index]) if invalid else values
        for index, values in enumerate(
            (backend.gas_density, backend.gas_mu, backend.gas_temperature)
        )
    )
    flow = (backend.ux, backend.uy, backend.uz)[:dim]
    point = (0.2, 0.3, 0.4)
    position = (*point[:dim], *((0.0,) * (3 - dim)))
    body = tuple(0.1 * (index + 1) for index in range(dim))
    extras = tuple(np.asarray([0.05]) for _ in range(dim))
    payload = [
        _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
        *body,
        1.1,
        1.7e-5,
        290.0,
        *backend.axes,
        backend.times,
        *flow,
    ]
    payload += [*extras, True, *thermodynamics, backend.valid_mask]
    stage = [0, 0.25, *position, 0.1, -0.1, 0.05, *_PARTICLE_STAGE, *payload]
    return stage, (*position, *payload), point[:dim]


@pytest.mark.parametrize(
    ("kernel", "dim", "suffix"),
    [(grid2d_kernel, 2, "bilinear"), (grid3d_kernel, 3, "trilinear")],
)
@pytest.mark.parametrize("invalid", [False, True])
def test_regular_grid_python_leaves_match_compiled(
    monkeypatch,
    kernel,
    dim: int,
    suffix: str,
    invalid: bool,
) -> None:
    backend = _regular_backend(dim)
    stage_args, support_args, point = _regular_leaf_arguments(backend, dim, invalid)
    stage = getattr(kernel, f"_regular_{dim}d_stage")
    support = getattr(kernel, f"_regular_{dim}d_support")
    sample_time = getattr(kernel, f"_sample_time_{suffix}")
    sample_args = [
        (backend.ux, backend.times, *backend.axes, time_s, *point)
        for time_s in (-0.5, 0.25, 1.5)
    ]
    compiled = (
        stage(*stage_args),
        support(*support_args),
        [sample_time(*args) for args in sample_args],
    )

    _use_python_dispatchers(monkeypatch, mask_kernel, kernel)
    python = (
        stage.py_func(*stage_args),
        support.py_func(*support_args),
        [sample_time.py_func(*args) for args in sample_args],
    )

    for actual, expected in zip(python, compiled, strict=True):
        np.testing.assert_array_equal(actual, expected)


def _triangle_leaf_arguments(backend):
    mesh = (
        backend.mesh_vertices,
        backend.mesh_triangles,
        backend.accel_origin,
        backend.accel_cell_size,
        *backend.accel_shape,
        backend.accel_cell_offsets,
        backend.accel_triangle_indices,
        backend.support_tolerance_m,
    )
    fields = tuple(
        getattr(backend, name)
        for name in ("ux", "uy", "gas_density", "gas_mu", "gas_temperature")
    )
    payload = [
        _EPSTEIN_DEFAULT_ACCOMMODATION_DELTA,
        0.1,
        -0.2,
        *mesh,
        backend.times,
        *fields,
    ]
    payload += [np.asarray([0.05]), np.asarray([-0.05])]
    return mesh, fields, payload


def _triangle_find_arguments(
    candidate_order: tuple[int, int] = (0, 1),
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    np.ndarray,
    np.ndarray,
    float,
]:
    return (
        np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        ),
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        np.asarray([0.0, 0.0], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64),
        1,
        1,
        np.asarray([0, 2], dtype=np.int32),
        np.asarray(candidate_order, dtype=np.int32),
        1.0e-12,
    )


def test_triangle_finder_preserves_dispatcher_signature_strict_tie_and_dtypes() -> None:
    finder = triangle_kernel._find_triangle_and_barycentric
    python_finder = vars(finder)["py_func"]
    arguments = _triangle_find_arguments((1, 0))

    assert tuple(inspect.signature(python_finder).parameters) == (
        "vertices",
        "triangles",
        "accel_origin",
        "accel_cell_size",
        "accel_nx",
        "accel_ny",
        "accel_cell_offsets",
        "accel_triangle_indices",
        "support_tolerance",
        "x",
        "y",
    )
    assert arguments[0].dtype == arguments[2].dtype == np.float64
    assert arguments[1].dtype == arguments[6].dtype == arguments[7].dtype == np.int32
    compiled = finder(*arguments, 0.5, 0.5)
    python = python_finder(*arguments, 0.5, 0.5)

    np.testing.assert_array_equal(compiled, python)
    assert compiled == (1, 0.5, 0.5, 0.0)
    assert getattr(finder, "nopython_signatures", ())


def test_triangle_finder_skips_zero_area_and_collinear_candidates_in_order() -> None:
    finder = triangle_kernel._find_triangle_and_barycentric
    python_finder = vars(finder)["py_func"]
    arguments = (
        np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]],
            dtype=np.float64,
        ),
        np.asarray([[0, 0, 0], [0, 1, 2], [0, 1, 3]], dtype=np.int32),
        np.asarray([0.0, 0.0], dtype=np.float64),
        np.asarray([2.0, 1.0], dtype=np.float64),
        1,
        1,
        np.asarray([0, 3], dtype=np.int32),
        np.asarray([0, 1, 2], dtype=np.int32),
        1.0e-12,
    )

    compiled = finder(*arguments, 0.25, 0.25)
    python = python_finder(*arguments, 0.25, 0.25)

    np.testing.assert_array_equal(compiled, python)
    assert compiled == (2, 0.5, 0.25, 0.25)


@settings(max_examples=32, deadline=None)
@given(
    x=st.floats(
        min_value=-0.25,
        max_value=1.25,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
    y=st.floats(
        min_value=-0.25,
        max_value=1.25,
        allow_nan=False,
        allow_infinity=False,
        width=64,
    ),
)
def test_triangle_finder_compiled_and_python_paths_are_bit_exact(
    x: float,
    y: float,
) -> None:
    finder = triangle_kernel._find_triangle_and_barycentric
    python_finder = vars(finder)["py_func"]
    arguments = _triangle_find_arguments()

    np.testing.assert_array_equal(
        finder(*arguments, x, y),
        python_finder(*arguments, x, y),
    )


def test_triangle_support_preserves_clean_and_hard_invalid_status_codes() -> None:
    backend = _triangle_backend()
    _mesh, _fields, payload = _triangle_leaf_arguments(backend)
    support = triangle_kernel._triangle_2d_support

    assert support(0.2, 0.3, 0.0, *payload) == VALID_MASK_STATUS_CLEAN
    assert support(-0.1, 0.2, 0.0, *payload) == VALID_MASK_STATUS_HARD_INVALID
    assert getattr(support, "nopython_signatures", ())


def test_triangle_python_leaves_match_compiled(monkeypatch) -> None:
    backend = _triangle_backend()
    mesh, fields, payload = _triangle_leaf_arguments(backend)
    points = ((0.2, 0.3), (-0.1, 0.2))
    find_args = [(*mesh, *point) for point in points]
    series_args = [
        (values, times, backend.mesh_triangles, 0, 0.5, 0.25, 0.25, time_s)
        for values, times, time_s in (
            (backend.ux[0], backend.times, 0.25),
            (backend.ux[:1], backend.times[:1], 0.25),
            (backend.ux, backend.times, -0.5),
            (backend.ux, backend.times, 0.25),
            (backend.ux, backend.times, 1.5),
        )
    ]
    flow_args = [(*mesh, backend.times, *fields, 0.25, *point) for point in points]
    position = (*points[0], 0.0)
    stage_args = [0, 0.25, *position, 0.1, -0.1, 0.0, *_PARTICLE_STAGE, *payload]
    dispatchers = (
        triangle_kernel._find_triangle_and_barycentric,
        triangle_kernel._sample_triangle_vertex_series,
        triangle_kernel._sample_triangle_mesh_flow,
        triangle_kernel._triangle_2d_stage,
        triangle_kernel._triangle_2d_support,
    )
    arguments = (
        find_args,
        series_args,
        flow_args,
        [stage_args],
        [[*position, *payload]],
    )
    compiled = [
        [function(*args) for args in cases]
        for function, cases in zip(dispatchers, arguments, strict=True)
    ]
    python_functions = tuple(vars(function)["py_func"] for function in dispatchers)

    _use_python_dispatchers(monkeypatch, triangle_kernel)
    python = [
        [function(*args) for args in cases]
        for function, cases in zip(python_functions, arguments, strict=True)
    ]

    for actual, expected in zip(python, compiled, strict=True):
        np.testing.assert_array_equal(actual, expected)
