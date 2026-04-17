from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import matplotlib
import numpy as np
import pandas as pd
import pytest
import yaml

matplotlib.use('Agg', force=True)

from particle_tracer_unified.core.geometry2d import (
    build_boundary_loops_2d,
    point_inside_boundary_edges_2d_with_boundary,
    points_inside_boundary_loops_2d,
    points_inside_boundary_loops_2d_with_boundary,
    validate_boundary_edges_2d,
)
from particle_tracer_unified.core.boundary_service import (
    BoundaryHit,
    build_boundary_service,
    nearest_boundary_edge_features_2d,
    normalize_polyline_alpha,
    polyline_hit_from_boundary_edges,
    polyline_hits_from_boundary_edges_batch,
    segment_hit_from_loop_bisection,
)
from particle_tracer_unified.core.field_backend import field_backend_kind, field_backend_report, sample_field_valid_status
from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
    choose_velocity_quantity_names,
    sample_quantity_series,
    sample_valid_mask_status,
)
from particle_tracer_unified.core.geometry3d import build_triangle_surface, point_inside_surface, validate_closed_surface_triangles
from particle_tracer_unified.core.grid_sampling import locate_axis_interval, sample_grid_scalar
from particle_tracer_unified.core.triangle_mesh_sampling_2d import (
    build_triangle_candidate_grid,
    locate_triangle_containing_point,
    sample_triangle_mesh_series,
    sample_triangle_mesh_status,
)
from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    GeometryND,
    GeometryProviderND,
    ProcessStepRow,
    QuantitySeriesND,
    RegularFieldND,
    TriangleMeshField2D,
)
from particle_tracer_unified.core.integrator_registry import get_integrator_spec, integrator_spec_from_mode
from particle_tracer_unified.core.input_contract import build_initial_particle_field_support_report
from particle_tracer_unified.core.provider_contract import build_boundary_field_support_report
from particle_tracer_unified.solvers.integrator_common import (
    DRAG_MODEL_SCHILLER_NAUMANN,
    DRAG_MODEL_STOKES,
    drag_model_mode_from_name,
    effective_tau_from_slip_speed,
)
from particle_tracer_unified.core.source_registry import get_source_law
from particle_tracer_unified.core.source_resolution import global_source_defaults
from particle_tracer_unified.io.runtime_builder import build_prepared_runtime_from_yaml, build_runtime_from_config
from particle_tracer_unified.io.tables import (
    load_materials_csv,
    load_part_walls_csv,
    load_process_steps_csv,
    load_source_events_csv,
)
from particle_tracer_unified.providers.precomputed import build_precomputed_field, build_precomputed_geometry, build_precomputed_triangle_mesh_field
from particle_tracer_unified.providers.synthetic import build_synthetic_field
from particle_tracer_unified.solvers.high_fidelity_collision import (
    WallHitStepResult,
    _apply_wall_hit_step,
    _classify_trial_collisions,
    _advance_colliding_particle,
)
from particle_tracer_unified.solvers.high_fidelity_freeflight import (
    RegularRectilinearCompiledBackend,
    TriangleMesh2DCompiledBackend,
    ValidMaskPrefixResolution,
    _advance_trial_particles,
    _compile_runtime_arrays,
)
from particle_tracer_unified.solvers.compiled_field_backend import (
    compiled_gas_property_report,
    sample_compiled_acceleration_vector,
    sample_compiled_acceleration_vectors,
    sample_compiled_flow_vector,
    sample_compiled_flow_vectors,
    sample_compiled_gas_properties,
    sample_compiled_valid_mask_status,
    sample_compiled_valid_mask_statuses,
)
from particle_tracer_unified.solvers.high_fidelity_runtime import (
    _apply_valid_mask_retry_then_stop,
    _advance_contact_sliding_particles_2d,
    _advance_contact_sliding_particles_3d,
    _initial_collision_diagnostics,
    RuntimeState,
    SolverRuntimeOptions,
    run_prepared_runtime,
)
from particle_tracer_unified.solvers.runtime_outputs import RuntimeOutputOptions
from particle_tracer_unified.solvers.solver_entrypoints import (
    build_prepared_runtime_for_dim,
    build_prepared_runtime_2d,
    build_prepared_runtime_3d,
    run_solver_2d_from_yaml,
    run_solver_3d_from_yaml,
)
from tools.build_comsol_case import (
    _merge_near_duplicate_axis,
    _order_quad_vertices,
    _points_inside_quads,
    _sample_points_in_quads,
    build_precomputed_arrays,
    parse_comsol_mphtxt,
    write_particles_for_case,
    write_case_files,
)
from tools.compare_against_reference import (
    class_match_ratio,
    class_transition_summary,
    geometry_feature_delta_summary,
    main as compare_against_reference_main,
)
from run_from_yaml import main as run_from_yaml_main
from tools.export_boundary_diagnostics_visuals import export_boundary_diagnostics
from tools.export_trajectory_animation import _interpolate_frames, _prepare_event_overlay
from tools.export_result_graphs import export_result_graphs
from tools.export_visualizations import export_visualizations
from tools.state_contract import classify_particle_states, particle_class_frame
from tools.visualization_common import final_state_counts, state_labels, step_state_count_series

ROOT = Path(__file__).resolve().parents[1]
_collision_diag_stub = _initial_collision_diagnostics


def _absolutize_paths(cfg: Mapping[str, Any], base_dir: Path) -> None:
    paths = cfg.get('paths', {})
    if not isinstance(paths, dict):
        paths = {}
    for key, value in list(paths.items()):
        if value is None or str(value).strip() == '':
            continue
        p = Path(str(value))
        paths[key] = str((base_dir / p).resolve() if not p.is_absolute() else p)
    providers = cfg.get('providers', {})
    if not isinstance(providers, dict):
        return
    for provider_name in ('geometry', 'field'):
        provider_cfg = providers.get(provider_name, {})
        if not isinstance(provider_cfg, dict):
            continue
        npz_path = provider_cfg.get('npz_path')
        if npz_path is None or str(npz_path).strip() == '':
            continue
        p = Path(str(npz_path))
        provider_cfg['npz_path'] = str((base_dir / p).resolve() if not p.is_absolute() else p)


def _write_config(tmp_path: Path, template: Path, mutate=None) -> Path:
    payload = yaml.safe_load(template.read_text(encoding='utf-8')) or {}
    if mutate is not None:
        mutate(payload)
    _absolutize_paths(payload, template.parent)
    out = tmp_path / 'run_config.yaml'
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')
    return out


def _write_field_bundle(path: Path, axes_x: np.ndarray, axes_y: np.ndarray, *, axis_0_shift: float = 0.0, include_wall_quantities: bool = True) -> Path:
    shape = (axes_x.size, axes_y.size)
    xx, yy = np.meshgrid(axes_x, axes_y, indexing='ij')
    payload: dict[str, Any] = {
        'axis_0': np.asarray(axes_x + float(axis_0_shift), dtype=np.float64),
        'axis_1': np.asarray(axes_y, dtype=np.float64),
        'times': np.asarray([0.0], dtype=np.float64),
        'valid_mask': np.ones(shape, dtype=bool),
        'ux': 0.1 * np.ones(shape, dtype=np.float64),
        'uy': 0.05 * np.cos(xx * 10.0) * np.ones(shape, dtype=np.float64),
    }
    if include_wall_quantities:
        payload['mu'] = 1.8e-5 * np.ones(shape, dtype=np.float64)
        payload['tauw'] = 0.2 + 0.0 * xx
        payload['u_tau'] = 0.4 + 0.0 * yy
    np.savez_compressed(path, **payload)
    return path


def _write_precomputed_geometry_npz(path: Path, axes_x: np.ndarray, axes_y: np.ndarray, *, valid_mask: np.ndarray) -> Path:
    xx, yy = np.meshgrid(axes_x, axes_y, indexing='ij')
    sdf = np.where(np.asarray(valid_mask, dtype=bool), -0.1, 0.1).astype(np.float64)
    np.savez_compressed(
        path,
        axis_0=np.asarray(axes_x, dtype=np.float64),
        axis_1=np.asarray(axes_y, dtype=np.float64),
        sdf=sdf,
        normal_0=np.zeros_like(xx, dtype=np.float64),
        normal_1=np.ones_like(yy, dtype=np.float64),
        valid_mask=np.asarray(valid_mask, dtype=bool),
        nearest_boundary_part_id_map=np.ones_like(valid_mask, dtype=np.int32),
    )
    return path


def _regular_field_provider_from_arrays(
    axes: tuple[np.ndarray, ...],
    valid_mask: np.ndarray,
    quantities: Mapping[str, np.ndarray],
) -> FieldProviderND:
    quantity_series = {
        name: QuantitySeriesND(
            name=name,
            unit='',
            times=np.asarray([0.0], dtype=np.float64),
            data=np.asarray(values, dtype=np.float64),
            metadata={},
        )
        for name, values in quantities.items()
    }
    field = RegularFieldND(
        spatial_dim=len(axes),
        coordinate_system='cartesian_xy' if len(axes) == 2 else 'cartesian_xyz',
        axis_names=tuple('xyz'[: len(axes)]),
        axes=tuple(np.asarray(axis, dtype=np.float64) for axis in axes),
        quantities=quantity_series,
        valid_mask=np.asarray(valid_mask, dtype=bool),
        time_mode='steady',
        metadata={'provider_kind': 'precomputed_npz'},
    )
    return FieldProviderND(field=field, kind='precomputed_npz')


def _geometry_provider_from_arrays(
    axes: tuple[np.ndarray, ...],
    valid_mask: np.ndarray,
    sdf: np.ndarray,
    normal_components: tuple[np.ndarray, ...],
) -> GeometryProviderND:
    geometry = GeometryND(
        spatial_dim=len(axes),
        coordinate_system='cartesian_xy' if len(axes) == 2 else 'cartesian_xyz',
        axes=tuple(np.asarray(axis, dtype=np.float64) for axis in axes),
        valid_mask=np.asarray(valid_mask, dtype=bool),
        sdf=np.asarray(sdf, dtype=np.float64),
        normal_components=tuple(np.asarray(comp, dtype=np.float64) for comp in normal_components),
        nearest_boundary_part_id_map=np.ones_like(np.asarray(valid_mask, dtype=bool), dtype=np.int32),
        source_kind='synthetic',
        metadata={},
    )
    return GeometryProviderND(geometry=geometry, kind='synthetic')


def _write_triangle_mesh_field_npz(path: Path) -> Path:
    vertices = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    times = np.asarray([0.0, 1.0], dtype=np.float64)
    ux_t0 = vertices[:, 0] + 2.0 * vertices[:, 1]
    ux_t1 = ux_t0 + 1.0
    uy_t0 = 3.0 * vertices[:, 0] - vertices[:, 1]
    uy_t1 = uy_t0 - 0.5
    payload = {
        'mesh_vertices': vertices,
        'mesh_triangles': triangles,
        'times': times,
        'ux': np.stack([ux_t0, ux_t1], axis=0),
        'uy': np.stack([uy_t0, uy_t1], axis=0),
        'mu': np.stack([1.8e-5 * np.ones(vertices.shape[0], dtype=np.float64) for _ in times], axis=0),
        'tauw': np.stack([0.5 * np.ones(vertices.shape[0], dtype=np.float64) for _ in times], axis=0),
        'u_tau': np.stack([0.3 * np.ones(vertices.shape[0], dtype=np.float64) for _ in times], axis=0),
        'metadata_json': np.asarray(
            json.dumps(
                {
                    'provider_kind': 'precomputed_triangle_mesh_npz',
                    'field_backend_kind': 'triangle_mesh_2d',
                    'support_tolerance_m': 2.0e-6,
                }
            )
        ),
    }
    np.savez_compressed(path, **payload)
    return path


def _cube_triangles_oriented() -> np.ndarray:
    corners = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    tri_vertex_ids = [
        (0, 2, 1), (0, 3, 2),
        (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4),
        (1, 2, 6), (1, 6, 5),
        (3, 6, 2), (3, 7, 6),
        (0, 7, 3), (0, 4, 7),
    ]
    return np.asarray([[corners[a], corners[b], corners[c]] for a, b, c in tri_vertex_ids], dtype=np.float64)


def test_process_steps_csv_minimal_overlay_is_loaded():
    table = load_process_steps_csv(ROOT / 'schemas' / 'process_steps.example.csv')
    assert len(table.rows) == 1
    assert table.rows[0].step_name == 'run'
    assert table.active_at(0.5) is table.rows[0]


def test_grid_sampling_uses_shared_linear_interpolation_contract():
    axis = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    lo, hi, alpha = locate_axis_interval(axis, 0.25)
    assert (int(lo), int(hi)) == (0, 1)
    assert float(alpha) == pytest.approx(0.25)

    arr2 = np.asarray(
        [
            [0.0, 1.0, 2.0],
            [10.0, 11.0, 12.0],
            [20.0, 21.0, 22.0],
        ],
        dtype=np.float64,
    )
    val2 = sample_grid_scalar(arr2, (axis, axis), np.asarray([0.5, 0.5], dtype=np.float64))
    assert float(val2) == pytest.approx(5.5)

    arr3 = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    val3 = sample_grid_scalar(arr3, (axis, axis, axis), np.asarray([0.5, 0.5, 0.5], dtype=np.float64))
    corners = arr3[:2, :2, :2]
    assert float(val3) == pytest.approx(float(np.mean(corners)))


def test_valid_mask_sampling_treats_points_outside_axes_as_hard_invalid():
    axes = np.asarray([0.0, 1.0], dtype=np.float64)
    mask = np.ones((2, 2), dtype=bool)

    status = sample_valid_mask_status(mask, (axes, axes), np.asarray([-0.1, 0.5], dtype=np.float64))

    assert int(status) == int(VALID_MASK_STATUS_HARD_INVALID)


def test_schiller_naumann_drag_reduces_effective_relaxation_time_for_finite_re():
    tau_stokes = 1.0
    slip_speed = 50.0
    diameter = 1.0e-4
    gas_density = 1.2
    gas_mu = 1.8e-5

    stokes_tau = effective_tau_from_slip_speed(
        tau_stokes,
        slip_speed,
        diameter,
        gas_density,
        gas_mu,
        DRAG_MODEL_STOKES,
        1.0e-9,
    )
    finite_re_tau = effective_tau_from_slip_speed(
        tau_stokes,
        slip_speed,
        diameter,
        gas_density,
        gas_mu,
        DRAG_MODEL_SCHILLER_NAUMANN,
        1.0e-9,
    )

    assert drag_model_mode_from_name('schiller-naumann') == int(DRAG_MODEL_SCHILLER_NAUMANN)
    assert float(stokes_tau) == pytest.approx(tau_stokes)
    assert 0.0 < float(finite_re_tau) < float(stokes_tau)


def test_field_sampling_shared_helpers_cover_velocity_name_resolution_and_transient_sampling():
    field = SimpleNamespace(
        spatial_dim=2,
        coordinate_system='axisymmetric_rz',
        quantities={'ur': object(), 'uz': object()},
    )
    assert choose_velocity_quantity_names(field, 2) == ('ur', 'uz')

    series = SimpleNamespace(
        times=np.asarray([0.0, 1.0], dtype=np.float64),
        data=np.asarray(
            [
                [[0.0, 10.0], [20.0, 30.0]],
                [[100.0, 110.0], [120.0, 130.0]],
            ],
            dtype=np.float64,
        ),
    )
    axes = (np.asarray([0.0, 1.0], dtype=np.float64), np.asarray([0.0, 1.0], dtype=np.float64))
    value = sample_quantity_series(series, axes, np.asarray([0.5, 0.5], dtype=np.float64), 0.25, mode='linear')
    expected_t0 = 15.0
    expected_t1 = 115.0
    assert float(value) == pytest.approx(expected_t0 * 0.75 + expected_t1 * 0.25)


def test_sample_valid_mask_status_distinguishes_clean_mixed_and_hard_invalid():
    axes = np.asarray([0.0, 1.0], dtype=np.float64)
    mask = np.asarray([[1, 1], [1, 0]], dtype=bool)

    clean_status = sample_valid_mask_status(np.ones((2, 2), dtype=bool), (axes, axes), np.asarray([0.5, 0.5], dtype=np.float64))
    mixed_status = sample_valid_mask_status(mask, (axes, axes), np.asarray([0.2, 0.2], dtype=np.float64))
    hard_status = sample_valid_mask_status(mask, (axes, axes), np.asarray([0.9, 0.9], dtype=np.float64))

    assert int(clean_status) == int(VALID_MASK_STATUS_CLEAN)
    assert int(mixed_status) == int(VALID_MASK_STATUS_MIXED_STENCIL)
    assert int(hard_status) == int(VALID_MASK_STATUS_HARD_INVALID)


def test_triangle_mesh_sampling_helpers_resolve_containment_and_barycentric_interpolation():
    vertices = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    accel_origin, accel_cell_size, accel_shape, accel_offsets, accel_triangle_indices = build_triangle_candidate_grid(vertices, triangles)
    point_inside = np.asarray([0.75, 0.25], dtype=np.float64)
    point_outside = np.asarray([1.25, 0.25], dtype=np.float64)
    tri_idx, bary = locate_triangle_containing_point(
        vertices=vertices,
        triangles=triangles,
        accel_origin=accel_origin,
        accel_cell_size=accel_cell_size,
        accel_shape=accel_shape,
        accel_cell_offsets=accel_offsets,
        accel_triangle_indices=accel_triangle_indices,
        position=point_inside,
    )
    assert int(tri_idx) >= 0
    assert float(np.sum(bary)) == pytest.approx(1.0, abs=1e-12)

    field = TriangleMeshField2D(
        spatial_dim=2,
        coordinate_system='cartesian_xy',
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        quantities={
            'ux': QuantitySeriesND(
                name='ux',
                unit='m/s',
                times=np.asarray([0.0, 1.0], dtype=np.float64),
                data=np.asarray(
                    [
                        vertices[:, 0] + 2.0 * vertices[:, 1],
                        vertices[:, 0] + 2.0 * vertices[:, 1] + 1.0,
                    ],
                    dtype=np.float64,
                ),
                metadata={},
            )
        },
        accel_origin=accel_origin,
        accel_cell_size=accel_cell_size,
        accel_shape=accel_shape,
        accel_cell_offsets=accel_offsets,
        accel_triangle_indices=accel_triangle_indices,
        time_mode='transient',
        metadata={'field_backend_kind': 'triangle_mesh_2d'},
    )
    value = sample_triangle_mesh_series(field.quantities['ux'], field, point_inside, 0.5, mode='linear')
    assert float(value) == pytest.approx(1.75, abs=1e-12)
    assert int(sample_triangle_mesh_status(field, point_inside)) == int(VALID_MASK_STATUS_CLEAN)
    assert int(sample_triangle_mesh_status(field, point_outside)) == int(VALID_MASK_STATUS_HARD_INVALID)


def test_precomputed_triangle_mesh_field_loader_reports_inside_clean_and_outside_hard_invalid(tmp_path: Path):
    mesh_path = _write_triangle_mesh_field_npz(tmp_path / 'field_mesh.npz')
    provider = build_precomputed_triangle_mesh_field(
        {'npz_path': str(mesh_path)},
        spatial_dim=2,
        coordinate_system='cartesian_xy',
        gas_density_kgm3=1.2,
    )
    assert field_backend_kind(provider) == 'triangle_mesh_2d'
    inside = sample_field_valid_status(provider, np.asarray([0.25, 0.25], dtype=np.float64))
    outside = sample_field_valid_status(provider, np.asarray([1.25, 0.25], dtype=np.float64))
    assert int(inside) == int(VALID_MASK_STATUS_CLEAN)
    assert int(outside) == int(VALID_MASK_STATUS_HARD_INVALID)


def test_precomputed_field_loader_rejects_nonfinite_values_inside_support(tmp_path: Path):
    axes = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    valid_mask = np.ones((3, 3), dtype=bool)
    ux = np.ones((3, 3), dtype=np.float64)
    ux[1, 1] = np.nan
    field_path = tmp_path / 'bad_field_values.npz'
    np.savez_compressed(
        field_path,
        axis_0=axes,
        axis_1=axes,
        times=np.asarray([0.0], dtype=np.float64),
        valid_mask=valid_mask,
        ux=ux,
        uy=np.zeros((3, 3), dtype=np.float64),
    )
    with pytest.raises(ValueError, match='inside field valid_mask support'):
        build_precomputed_field(
            {'npz_path': str(field_path)},
            spatial_dim=2,
            coordinate_system='cartesian_xy',
            axes=(axes, axes),
        )


def test_precomputed_triangle_mesh_field_loader_rejects_invalid_mesh_contract(tmp_path: Path):
    mesh_path = _write_triangle_mesh_field_npz(tmp_path / 'bad_mesh_field.npz')
    with np.load(mesh_path) as payload:
        data = {key: np.asarray(payload[key]) for key in payload.files}
    data['mesh_triangles'] = np.asarray([[0, 1, 99]], dtype=np.int32)
    np.savez_compressed(mesh_path, **data)
    with pytest.raises(ValueError, match='outside mesh_vertices'):
        build_precomputed_triangle_mesh_field(
            {'npz_path': str(mesh_path)},
            spatial_dim=2,
            coordinate_system='cartesian_xy',
        )


def test_compile_runtime_arrays_returns_regular_rectilinear_backend():
    axes = (
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )
    valid_mask = np.ones((3, 3), dtype=bool)
    field_provider = _regular_field_provider_from_arrays(
        axes,
        valid_mask,
        quantities={
            'ux': np.full((3, 3), 2.0, dtype=np.float64),
            'uy': np.full((3, 3), -1.0, dtype=np.float64),
        },
    )
    geometry_provider = _geometry_provider_from_arrays(
        axes,
        valid_mask,
        sdf=-np.ones((3, 3), dtype=np.float64),
        normal_components=(
            np.zeros((3, 3), dtype=np.float64),
            np.ones((3, 3), dtype=np.float64),
        ),
    )
    runtime = SimpleNamespace(geometry_provider=geometry_provider, field_provider=field_provider)
    compiled = _compile_runtime_arrays(runtime, spatial_dim=2)

    assert isinstance(compiled, RegularRectilinearCompiledBackend)
    assert compiled.backend_kind == 'regular_rectilinear'
    assert compiled.valid_mask.shape == (3, 3)
    assert compiled.core_valid_mask.shape == (3, 3)


def test_compile_runtime_arrays_samples_electric_force_from_particle_q_over_m_2d():
    axes = (
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )
    valid_mask = np.ones((3, 3), dtype=bool)
    ex_grid = np.stack(
        [
            np.full((3, 3), 2.0, dtype=np.float64),
            np.full((3, 3), 6.0, dtype=np.float64),
        ],
        axis=0,
    )
    ey_grid = np.zeros((2, 3, 3), dtype=np.float64)
    times = np.asarray([0.0, 1.0], dtype=np.float64)
    quantities = {
        'ux': np.zeros((2, 3, 3), dtype=np.float64),
        'uy': np.zeros((2, 3, 3), dtype=np.float64),
        'E_x': ex_grid,
        'E_y': ey_grid,
    }
    field_provider = FieldProviderND(
        field=RegularFieldND(
            spatial_dim=2,
            coordinate_system='cartesian_xy',
            axis_names=('x', 'y'),
            axes=axes,
            quantities={
                name: QuantitySeriesND(name=name, unit='', times=times, data=value, metadata={})
                for name, value in quantities.items()
            },
            valid_mask=valid_mask,
            time_mode='transient',
            metadata={'provider_kind': 'precomputed_npz'},
        ),
        kind='precomputed_npz',
    )
    geometry_provider = _geometry_provider_from_arrays(
        axes,
        valid_mask,
        sdf=-np.ones((3, 3), dtype=np.float64),
        normal_components=(np.zeros((3, 3), dtype=np.float64), np.ones((3, 3), dtype=np.float64)),
    )
    runtime = SimpleNamespace(geometry_provider=geometry_provider, field_provider=field_provider)

    compiled = _compile_runtime_arrays(runtime, spatial_dim=2)
    accel = sample_compiled_acceleration_vector(
        compiled,
        2,
        0.25,
        np.asarray([0.5, 0.5], dtype=np.float64),
        electric_q_over_m=-0.5,
    )

    assert compiled.acceleration_source == 'particle_charge_electric_field'
    assert compiled.electric_field_names == ('E_x', 'E_y')
    assert accel.tolist() == pytest.approx([-1.5, 0.0])


def test_compile_runtime_arrays_uses_field_gas_properties_for_epstein_drag():
    axes = (
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )
    valid_mask = np.ones((3, 3), dtype=bool)
    times = np.asarray([0.0, 1.0], dtype=np.float64)
    transient = (2, 3, 3)
    quantities = {
        'ux': QuantitySeriesND('ux', 'm/s', times, np.zeros(transient, dtype=np.float64), {}),
        'uy': QuantitySeriesND('uy', 'm/s', times, np.zeros(transient, dtype=np.float64), {}),
        'rho_g': QuantitySeriesND(
            'rho_g',
            'kg/m^3',
            times,
            np.stack(
                [
                    np.full((3, 3), 2.0e-5, dtype=np.float64),
                    np.full((3, 3), 6.0e-5, dtype=np.float64),
                ],
                axis=0,
            ),
            {},
        ),
        'T': QuantitySeriesND(
            'T',
            'K',
            np.asarray([0.0], dtype=np.float64),
            np.full((3, 3), 420.0, dtype=np.float64),
            {},
        ),
        'mu': QuantitySeriesND(
            'mu',
            'Pa s',
            np.asarray([0.0], dtype=np.float64),
            np.full((3, 3), 2.2e-5, dtype=np.float64),
            {},
        ),
        'p': QuantitySeriesND(
            'p',
            'Pa',
            times,
            np.full(transient, 3.0, dtype=np.float64),
            {},
        ),
    }
    field_provider = FieldProviderND(
        field=RegularFieldND(
            spatial_dim=2,
            coordinate_system='cartesian_xy',
            axis_names=('x', 'y'),
            axes=axes,
            quantities=quantities,
            valid_mask=valid_mask,
            time_mode='transient',
            metadata={'provider_kind': 'precomputed_npz'},
        ),
        kind='precomputed_npz',
    )
    geometry_provider = _geometry_provider_from_arrays(
        axes,
        valid_mask,
        sdf=-np.ones((3, 3), dtype=np.float64),
        normal_components=(np.zeros((3, 3), dtype=np.float64), np.ones((3, 3), dtype=np.float64)),
    )
    runtime = SimpleNamespace(
        geometry_provider=geometry_provider,
        field_provider=field_provider,
        gas=SimpleNamespace(density_kgm3=1.0, dynamic_viscosity_Pas=1.8e-5, temperature=300.0),
    )

    compiled = _compile_runtime_arrays(runtime, spatial_dim=2)
    rho, mu, temp = sample_compiled_gas_properties(
        compiled,
        0.5,
        np.asarray([0.5, 0.5], dtype=np.float64),
        fallback_density_kgm3=1.0,
        fallback_mu_pas=1.8e-5,
        fallback_temperature_K=300.0,
    )
    report = compiled_gas_property_report(
        compiled,
        fallback_density_kgm3=1.0,
        fallback_mu_pas=1.8e-5,
        fallback_temperature_K=300.0,
        drag_model_name='epstein',
    )

    assert compiled.times.tolist() == pytest.approx([0.0, 1.0])
    assert compiled.gas_density_source == 'field:rho_g'
    assert compiled.gas_temperature_source == 'field:T'
    assert compiled.gas_mu_source == 'field:mu'
    assert rho == pytest.approx(4.0e-5)
    assert mu == pytest.approx(2.2e-5)
    assert temp == pytest.approx(420.0)
    assert report['uses_field_density'] == 1
    assert report['pressure_source'] == 'diagnostic_only_not_used_by_drag'


def test_field_backend_report_flags_quantity_time_axis_mismatch():
    axes = (
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )
    valid_mask = np.ones((3, 3), dtype=bool)
    ux_times = np.asarray([0.0, 1.0], dtype=np.float64)
    uy_times = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    field_provider = FieldProviderND(
        field=RegularFieldND(
            spatial_dim=2,
            coordinate_system='cartesian_xy',
            axis_names=('x', 'y'),
            axes=axes,
            quantities={
                'ux': QuantitySeriesND(
                    name='ux',
                    unit='m/s',
                    times=ux_times,
                    data=np.zeros((2, 3, 3), dtype=np.float64),
                    metadata={},
                ),
                'uy': QuantitySeriesND(
                    name='uy',
                    unit='m/s',
                    times=uy_times,
                    data=np.zeros((3, 3, 3), dtype=np.float64),
                    metadata={},
                ),
            },
            valid_mask=valid_mask,
            time_mode='transient',
            metadata={'provider_kind': 'precomputed_npz'},
        ),
        kind='precomputed_npz',
    )

    report = field_backend_report(field_provider)

    assert report['time_axis']['time_mode'] == 'transient'
    assert report['time_axis']['time_count'] == 2
    assert report['time_axis']['quantity_time_axis_reference'] == 'ux'
    assert report['time_axis']['quantity_time_axis_mismatch_count'] == 1
    assert report['time_axis']['quantity_time_axis_mismatches'] == ['uy']


def test_compile_runtime_arrays_rejects_solver_quantity_time_axis_mismatch():
    axes = (
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )
    valid_mask = np.ones((3, 3), dtype=bool)
    field_provider = FieldProviderND(
        field=RegularFieldND(
            spatial_dim=2,
            coordinate_system='cartesian_xy',
            axis_names=('x', 'y'),
            axes=axes,
            quantities={
                'ux': QuantitySeriesND(
                    name='ux',
                    unit='m/s',
                    times=np.asarray([0.0, 1.0], dtype=np.float64),
                    data=np.zeros((2, 3, 3), dtype=np.float64),
                    metadata={},
                ),
                'uy': QuantitySeriesND(
                    name='uy',
                    unit='m/s',
                    times=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
                    data=np.zeros((3, 3, 3), dtype=np.float64),
                    metadata={},
                ),
            },
            valid_mask=valid_mask,
            time_mode='transient',
            metadata={'provider_kind': 'precomputed_npz'},
        ),
        kind='precomputed_npz',
    )
    geometry_provider = _geometry_provider_from_arrays(
        axes,
        valid_mask,
        sdf=-np.ones((3, 3), dtype=np.float64),
        normal_components=(np.zeros((3, 3), dtype=np.float64), np.ones((3, 3), dtype=np.float64)),
    )
    runtime = SimpleNamespace(geometry_provider=geometry_provider, field_provider=field_provider)

    with pytest.raises(ValueError, match='must share one time axis'):
        _compile_runtime_arrays(runtime, spatial_dim=2)


def test_synthetic_transient_field_requires_clean_time_axis():
    axes = (
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match='strictly increasing'):
        build_synthetic_field(
            {'kind': 'linear_shear', 'time_mode': 'transient', 'times': [0.0, 0.5, 0.5]},
            spatial_dim=2,
            coordinate_system='cartesian_xy',
            axes=axes,
        )

    with pytest.raises(ValueError, match='steady requires exactly one time value'):
        build_synthetic_field(
            {'kind': 'linear_shear', 'time_mode': 'steady', 'times': [0.0, 1.0]},
            spatial_dim=2,
            coordinate_system='cartesian_xy',
            axes=axes,
        )

    provider = build_synthetic_field(
        {'kind': 'linear_shear', 'time_mode': 'transient', 'times': [0.0, 0.5, 1.0]},
        spatial_dim=2,
        coordinate_system='cartesian_xy',
        axes=axes,
    )
    report = field_backend_report(provider)

    assert report['time_axis']['time_mode'] == 'transient'
    assert report['time_axis']['time_count'] == 3
    assert report['time_axis']['quantity_time_axis_mismatch_count'] == 0


def test_compiled_regular_backend_batch_sampling_matches_scalar_sampling_2d():
    axes = (
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )
    valid_mask = np.ones((3, 3), dtype=bool)
    valid_mask[2, 2] = False
    times = np.asarray([0.0, 1.0], dtype=np.float64)
    x_grid, y_grid = np.meshgrid(axes[0], axes[1], indexing='ij')
    quantities = {
        'ux': np.stack((x_grid, 2.0 * x_grid), axis=0),
        'uy': np.stack((y_grid, -y_grid), axis=0),
        'E_x': np.stack((x_grid + y_grid, 3.0 * (x_grid + y_grid)), axis=0),
        'E_y': np.stack((x_grid - y_grid, 0.5 * (x_grid - y_grid)), axis=0),
    }
    field_provider = FieldProviderND(
        field=RegularFieldND(
            spatial_dim=2,
            coordinate_system='cartesian_xy',
            axis_names=('x', 'y'),
            axes=axes,
            quantities={
                name: QuantitySeriesND(name=name, unit='', times=times, data=value, metadata={})
                for name, value in quantities.items()
            },
            valid_mask=valid_mask,
            time_mode='transient',
            metadata={'provider_kind': 'precomputed_npz'},
        ),
        kind='precomputed_npz',
    )
    geometry_provider = _geometry_provider_from_arrays(
        axes,
        valid_mask,
        sdf=-np.ones((3, 3), dtype=np.float64),
        normal_components=(np.zeros((3, 3), dtype=np.float64), np.ones((3, 3), dtype=np.float64)),
    )
    runtime = SimpleNamespace(geometry_provider=geometry_provider, field_provider=field_provider)
    compiled = _compile_runtime_arrays(runtime, spatial_dim=2)
    points = np.asarray([[0.25, 0.25], [0.75, 0.25], [1.0, 1.0]], dtype=np.float64)
    t_eval = 0.25

    flow_batch = sample_compiled_flow_vectors(compiled, 2, t_eval, points)
    qom = np.asarray([1.0, 0.5, -1.0], dtype=np.float64)
    accel_batch = sample_compiled_acceleration_vectors(compiled, 2, t_eval, points, electric_q_over_m=qom)
    status_batch = sample_compiled_valid_mask_statuses(compiled, points)

    flow_scalar = np.asarray([sample_compiled_flow_vector(compiled, 2, t_eval, point) for point in points])
    accel_scalar = np.asarray(
        [
            sample_compiled_acceleration_vector(compiled, 2, t_eval, point, electric_q_over_m=float(qom[i]))
            for i, point in enumerate(points)
        ]
    )
    status_scalar = np.asarray([sample_compiled_valid_mask_status(compiled, point) for point in points])

    assert flow_batch == pytest.approx(flow_scalar)
    assert accel_batch == pytest.approx(accel_scalar)
    assert status_batch.tolist() == status_scalar.tolist()


def test_compile_runtime_arrays_returns_triangle_mesh_backend(tmp_path: Path):
    mesh_path = _write_triangle_mesh_field_npz(tmp_path / 'field_mesh.npz')
    field_provider = build_precomputed_triangle_mesh_field(
        {'npz_path': str(mesh_path)},
        spatial_dim=2,
        coordinate_system='cartesian_xy',
        gas_density_kgm3=1.2,
    )
    axes = (
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )
    valid_mask = np.ones((3, 3), dtype=bool)
    geometry_provider = _geometry_provider_from_arrays(
        axes,
        valid_mask,
        sdf=-np.ones((3, 3), dtype=np.float64),
        normal_components=(
            np.zeros((3, 3), dtype=np.float64),
            np.ones((3, 3), dtype=np.float64),
        ),
    )
    runtime = SimpleNamespace(geometry_provider=geometry_provider, field_provider=field_provider)
    compiled = _compile_runtime_arrays(runtime, spatial_dim=2)

    assert isinstance(compiled, TriangleMesh2DCompiledBackend)
    assert compiled.backend_kind == 'triangle_mesh_2d'
    assert compiled.mesh_vertices.shape[1] == 2
    assert compiled.mesh_triangles.shape[1] == 3
    assert compiled.ux.ndim == 2
    assert compiled.uy.ndim == 2
    assert sample_compiled_acceleration_vector(compiled, 2, 0.5, np.asarray([0.25, 0.25], dtype=np.float64)).tolist() == pytest.approx([0.0, 0.0])


def test_trial_particle_advance_uses_particle_charge_electric_field_2d():
    spatial_dim = 2
    axes = tuple(np.asarray([0.0, 0.5, 1.0], dtype=np.float64) for _ in range(spatial_dim))
    valid_mask = np.ones(tuple(3 for _ in range(spatial_dim)), dtype=bool)
    quantities: dict[str, np.ndarray] = {
        'ux': np.zeros_like(valid_mask, dtype=np.float64),
        'uy': np.zeros_like(valid_mask, dtype=np.float64),
        'E_x': np.ones_like(valid_mask, dtype=np.float64) * 8.0,
        'E_y': np.zeros_like(valid_mask, dtype=np.float64),
    }
    field_provider = _regular_field_provider_from_arrays(axes, valid_mask, quantities=quantities)
    geometry_provider = _geometry_provider_from_arrays(
        axes,
        valid_mask,
        sdf=-np.ones_like(valid_mask, dtype=np.float64),
        normal_components=tuple(np.zeros_like(valid_mask, dtype=np.float64) for _ in range(spatial_dim)),
    )
    runtime = SimpleNamespace(geometry_provider=geometry_provider, field_provider=field_provider)
    compiled = _compile_runtime_arrays(runtime, spatial_dim=spatial_dim)
    x = np.asarray([[0.5] * spatial_dim], dtype=np.float64)
    v = np.zeros((1, spatial_dim), dtype=np.float64)
    active = np.asarray([True], dtype=bool)
    x_trial = np.zeros_like(x)
    v_trial = np.zeros_like(v)
    x_mid_trial = np.zeros_like(x)

    _advance_trial_particles(
        spatial_dim=spatial_dim,
        compiled=compiled,
        x=x,
        v=v,
        active=active,
        tau_p=np.asarray([1.0], dtype=np.float64),
        particle_diameter=np.asarray([1.0e-6], dtype=np.float64),
        flow_scale_particle=np.asarray([1.0], dtype=np.float64),
        drag_scale_particle=np.asarray([1.0], dtype=np.float64),
        body_scale_particle=np.asarray([1.0], dtype=np.float64),
        t=0.1,
        dt_step=0.1,
        phys={'flow_scale': 1.0, 'drag_tau_scale': 1.0, 'body_accel_scale': 1.0, 'min_tau_p_s': 1.0},
        body_accel=np.zeros(spatial_dim, dtype=np.float64),
        gas_density_kgm3=1.0,
        gas_mu_pas=1.8e-5,
        drag_model_mode=DRAG_MODEL_STOKES,
        integrator_mode=get_integrator_spec('drag_relaxation').mode,
        adaptive_substep_enabled=0,
        adaptive_substep_tau_ratio=0.5,
        adaptive_substep_max_splits=4,
        x_trial=x_trial,
        v_trial=v_trial,
        x_mid_trial=x_mid_trial,
        substep_counts=np.ones(1, dtype=np.int32),
        valid_mask_status_flags=np.zeros(1, dtype=np.uint8),
        electric_q_over_m_particle=np.asarray([1.0], dtype=np.float64),
    )

    assert v_trial[0, 0] > 0.0
    assert x_trial[0, 0] > x[0, 0]


def test_runtime_builder_preserves_provider_field_support_mask(tmp_path: Path):
    axes = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    field_valid_mask = np.asarray(
        [
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=bool,
    )
    geometry_valid_mask = np.asarray(
        [
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 0],
        ],
        dtype=bool,
    )
    expected_core_mask = field_valid_mask & geometry_valid_mask
    geom_path = _write_precomputed_geometry_npz(tmp_path / 'geom.npz', axes, axes, valid_mask=geometry_valid_mask)
    field_path = tmp_path / 'field.npz'
    payload = {
        'axis_0': axes,
        'axis_1': axes,
        'times': np.asarray([0.0], dtype=np.float64),
        'valid_mask': field_valid_mask,
        'ux': np.asarray(
            [
                [0.0, 1.0, 2.0],
                [10.0, 11.0, 12.0],
                [20.0, 21.0, 22.0],
            ],
            dtype=np.float64,
        ),
        'uy': np.zeros((3, 3), dtype=np.float64),
    }
    np.savez_compressed(field_path, **payload)

    cfg = yaml.safe_load((ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml').read_text(encoding='utf-8'))
    cfg['providers']['geometry'] = {'kind': 'precomputed_npz', 'npz_path': str(geom_path.resolve())}
    cfg['providers']['field'] = {'kind': 'precomputed_npz', 'npz_path': str(field_path.resolve())}
    _absolutize_paths(cfg, ROOT / 'examples' / 'minimal_2d')

    runtime = build_runtime_from_config(cfg, ROOT / 'examples' / 'minimal_2d')
    field = runtime.field_provider.field

    assert np.array_equal(np.asarray(field.valid_mask, dtype=bool), field_valid_mask)
    assert np.array_equal(np.asarray(field.core_valid_mask, dtype=bool), expected_core_mask)
    assert float(field.quantities['ux'].data[0, 2]) == pytest.approx(2.0)
    assert int(field.metadata['field_valid_node_count']) == int(np.count_nonzero(field_valid_mask))
    assert int(field.metadata['geometry_valid_node_count']) == int(np.count_nonzero(geometry_valid_mask))
    assert int(field.metadata['core_valid_node_count']) == int(np.count_nonzero(expected_core_mask))


def test_runtime_builder_rejects_recipe_manifest_path(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('paths', {}).update(
            {
                'recipe_manifest_yaml': str((tmp_path / 'recipe_manifest.yaml').resolve()),
            }
        ),
    )
    with pytest.raises(ValueError, match='recipe_manifest_yaml is no longer supported'):
        build_prepared_runtime_from_yaml(config_path)


def test_runtime_builder_allows_process_step_gaps_as_time_label_overlay(tmp_path: Path):
    steps = pd.DataFrame(
        [
            {'step_id': 1, 'step_name': 'etch', 'start_s': 0.0, 'end_s': 0.5},
            {'step_id': 2, 'step_name': 'purge', 'start_s': 0.75, 'end_s': 1.0},
        ]
    )
    steps_path = tmp_path / 'process_steps_gap_allowed.csv'
    steps.to_csv(steps_path, index=False)
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: (
            cfg.setdefault('paths', {}).update({'process_steps_csv': str(steps_path.resolve())}),
            cfg.setdefault('output', {}).update({'write_positions': 0, 'write_segmented_positions': 1}),
        ),
    )

    prepared = build_prepared_runtime_from_yaml(config_path)

    assert prepared.runtime.process_steps is not None
    assert [row.step_name for row in prepared.runtime.process_steps.rows] == ['etch', 'purge']
    assert prepared.runtime.process_steps.active_at(0.6) is None


def test_process_step_override_columns_are_rejected(tmp_path: Path):
    steps = pd.DataFrame(
        [
            {
                'step_id': 1,
                'step_name': 'run',
                'start_s': 0.0,
                'end_s': 0.1,
                'output_segment_name': 'run',
                'source_enabled': 0,
                'source_law_override': 'resuspension_shear_material',
                'wall_mode': 'stick',
                'output_save_positions': 0,
            }
        ]
    )
    steps_path = tmp_path / 'process_steps_with_override_columns.csv'
    steps.to_csv(steps_path, index=False)
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: (
            cfg.setdefault('paths', {}).update({'process_steps_csv': str(steps_path.resolve())}),
            cfg.setdefault('solver', {}).update({'t_end': 0.02, 'save_every': 1, 'plot_particle_limit': 0}),
        ),
    )

    with pytest.raises(ValueError, match='process_steps.csv supports only time-label columns'):
        build_prepared_runtime_from_yaml(config_path)


def test_runtime_builder_rejects_zero_duration_process_steps(tmp_path: Path):
    steps = pd.DataFrame(
        [
            {'step_id': 1, 'step_name': 'instant_marker', 'start_s': 0.5, 'end_s': 0.5},
        ]
    )
    steps_path = tmp_path / 'process_steps_zero_duration.csv'
    steps.to_csv(steps_path, index=False)
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('paths', {}).update({'process_steps_csv': str(steps_path.resolve())}),
    )
    with pytest.raises(ValueError, match='must have end_s > start_s'):
        build_prepared_runtime_from_yaml(config_path)


def test_runtime_builder_rejects_unresolved_source_event_bindings(tmp_path: Path):
    events = pd.DataFrame(
        [
            {
                'event_id': 1,
                'event_name': 'bad_binding',
                'event_kind': 'gaussian_burst',
                'enabled': 1,
                'bind_step_name': 'missing_step',
                'time_anchor': 'step_start',
            }
        ]
    )
    events_path = tmp_path / 'source_events_bad_binding.csv'
    events.to_csv(events_path, index=False)
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('paths', {}).update({'source_events_csv': str(events_path.resolve())}),
    )
    with pytest.raises(ValueError, match='Unresolved source event bindings'):
        build_prepared_runtime_from_yaml(config_path)


def test_source_events_reject_transition_bindings(tmp_path: Path):
    events_path = tmp_path / 'source_events_transition.csv'
    pd.DataFrame(
        [
            {
                'event_id': 1,
                'event_name': 'old_transition_binding',
                'event_kind': 'gaussian_burst',
                'enabled': 1,
                'center_s': 0.0,
                'sigma_s': 0.01,
                'amplitude': 1.0,
                'bind_transition_from': 'etch',
            }
        ]
    ).to_csv(events_path, index=False)

    with pytest.raises(ValueError, match='transition bindings are no longer supported'):
        load_source_events_csv(events_path)


def test_shared_source_schema_keeps_material_wall_loading_and_defaults_in_sync(tmp_path: Path):
    materials_path = tmp_path / 'materials_schema.csv'
    pd.DataFrame(
        [
            {
                'material_id': 1,
                'material_name': 'steel',
                'source_law_default': 'flake_burst_material',
                'source_speed_scale': 1.25,
                'source_resuspension_friction_velocity_threshold_mps': 0.12,
                'source_burst_max_factor': 3.5,
                'source_default_event_tag': 'material_evt',
                'wall_law': 'diffuse',
                'wall_restitution': 0.35,
                'critical_sticking_velocity_mps': 0.8,
                'physics_flow_scale': 1.1,
            }
        ]
    ).to_csv(materials_path, index=False)

    part_walls_path = tmp_path / 'part_walls_schema.csv'
    pd.DataFrame(
        [
            {
                'part_id': 10,
                'part_name': 'plate',
                'material_id': 1,
                'material_name': 'steel',
                'source_law': 'resuspension_shear_material',
                'source_resuspension_friction_velocity_threshold_mps': 0.33,
                'source_default_event_tag': 'wall_evt',
                'wall_law': 'stick',
                'critical_sticking_velocity_mps': 0.45,
                'physics_drag_tau_scale': 1.4,
            }
        ]
    ).to_csv(part_walls_path, index=False)

    materials = load_materials_csv(materials_path)
    walls = load_part_walls_csv(part_walls_path)
    defaults = global_source_defaults(
        {
            'default_law': 'thermal_reemission_source_material',
            'source_resuspension_friction_velocity_threshold_mps': 0.21,
            'source_burst_max_factor': 5.0,
        },
        gas_temperature=425.0,
        gas_viscosity=2.2e-5,
    )

    assert materials.rows[0].source_law == 'flake_burst_material'
    assert materials.rows[0].source_resuspension_utau_threshold_mps == pytest.approx(0.12)
    assert materials.rows[0].source_burst_max_factor == pytest.approx(3.5)
    assert materials.rows[0].source_default_event_tag == 'material_evt'
    assert materials.rows[0].wall_critical_sticking_velocity_mps == pytest.approx(0.8)
    assert materials.rows[0].physics_flow_scale == pytest.approx(1.1)

    assert walls.rows[0].source_law == 'resuspension_shear_material'
    assert walls.rows[0].source_resuspension_utau_threshold_mps == pytest.approx(0.33)
    assert walls.rows[0].source_default_event_tag == 'wall_evt'
    assert walls.rows[0].wall_critical_sticking_velocity_mps == pytest.approx(0.45)
    assert walls.rows[0].physics_drag_tau_scale == pytest.approx(1.4)

    assert defaults['source_law'] == 'thermal_reemission_source_material'
    assert defaults['source_temperature_K'] == pytest.approx(425.0)
    assert defaults['source_dynamic_viscosity_Pas'] == pytest.approx(2.2e-5)
    assert defaults['source_resuspension_utau_threshold_mps'] == pytest.approx(0.21)
    assert defaults['source_burst_max_factor'] == pytest.approx(5.0)
    assert get_source_law('flake_burst_material').parameters == (
        'source_speed_scale',
        'source_position_offset_m',
        'source_normal_speed_mean_mps',
        'source_normal_speed_std_mps',
        'source_tangent_speed_std_mps',
        'source_flake_weight',
        'source_burst_center_s',
        'source_burst_sigma_s',
        'source_burst_amplitude',
        'source_burst_period_s',
        'source_burst_phase_s',
        'source_burst_min_factor',
        'source_burst_max_factor',
    )


def test_class_match_ratio_compares_particle_end_states_by_particle_id():
    reference = pd.DataFrame(
        [
            {'particle_id': 1, 'active': 1, 'stuck': 0, 'absorbed': 0, 'escaped': 0},
            {'particle_id': 2, 'active': 0, 'stuck': 1, 'absorbed': 0, 'escaped': 0},
            {'particle_id': 3, 'active': 0, 'stuck': 0, 'absorbed': 1, 'escaped': 0},
            {'particle_id': 4, 'active': 0, 'stuck': 0, 'absorbed': 0, 'escaped': 1},
        ]
    )
    candidate = pd.DataFrame(
        [
            {'particle_id': 4, 'active': 0, 'stuck': 0, 'absorbed': 0, 'escaped': 1},
            {'particle_id': 3, 'active': 0, 'stuck': 0, 'absorbed': 1, 'escaped': 0},
            {'particle_id': 2, 'active': 1, 'stuck': 0, 'absorbed': 0, 'escaped': 0},
            {'particle_id': 1, 'active': 1, 'stuck': 0, 'absorbed': 0, 'escaped': 0},
        ]
    )
    ratio, compared = class_match_ratio(candidate, reference)
    assert compared == 4
    assert ratio == pytest.approx(0.75)


def test_class_match_ratio_uses_shared_particle_ids_only():
    reference = pd.DataFrame(
        [
            {'particle_id': 1, 'active': 1, 'stuck': 0, 'absorbed': 0, 'escaped': 0},
            {'particle_id': 2, 'active': 0, 'stuck': 1, 'absorbed': 0, 'escaped': 0},
        ]
    )
    candidate = pd.DataFrame(
        [
            {'particle_id': 2, 'active': 0, 'stuck': 1, 'absorbed': 0, 'escaped': 0},
            {'particle_id': 3, 'active': 0, 'stuck': 0, 'absorbed': 0, 'escaped': 1},
        ]
    )
    ratio, compared = class_match_ratio(candidate, reference)
    assert compared == 1
    assert ratio == pytest.approx(1.0)


def test_class_match_ratio_recognizes_invalid_mask_stopped_as_distinct_class():
    reference = pd.DataFrame(
        [
            {'particle_id': 1, 'active': 0, 'stuck': 0, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 1},
            {'particle_id': 2, 'active': 1, 'stuck': 0, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 0},
        ]
    )
    candidate = pd.DataFrame(
        [
            {'particle_id': 1, 'active': 1, 'stuck': 0, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 0},
            {'particle_id': 2, 'active': 1, 'stuck': 0, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 0},
        ]
    )
    ratio, compared = class_match_ratio(candidate, reference)
    assert compared == 2
    assert ratio == pytest.approx(0.5)


def test_class_transition_summary_reports_mismatched_end_states():
    reference = pd.DataFrame(
        [
            {'particle_id': 1, 'active': 0, 'stuck': 1, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 0},
            {'particle_id': 2, 'active': 0, 'stuck': 0, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 1},
            {'particle_id': 3, 'active': 1, 'stuck': 0, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 0},
        ]
    )
    candidate = pd.DataFrame(
        [
            {'particle_id': 1, 'active': 0, 'stuck': 0, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 1},
            {'particle_id': 2, 'active': 0, 'stuck': 1, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 0},
            {'particle_id': 3, 'active': 1, 'stuck': 0, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 0},
        ]
    )
    summary = class_transition_summary(candidate, reference)
    assert summary['compared_particles'] == 3
    assert summary['mismatch_count'] == 2
    assert {'reference_class': 'stuck', 'candidate_class': 'invalid_mask_stopped', 'count': 1} in summary['top_mismatches']
    assert {'reference_class': 'invalid_mask_stopped', 'candidate_class': 'stuck', 'count': 1} in summary['top_mismatches']


def test_geometry_feature_delta_summary_reports_sdf_and_distance_errors():
    axes = (np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 6))
    xx, _yy = np.meshgrid(axes[0], axes[1], indexing='ij')
    valid_mask = np.ones((6, 6), dtype=bool)
    sdf = 0.5 - xx
    geometry_provider = _geometry_provider_from_arrays(
        axes,
        valid_mask,
        sdf,
        (
            -np.ones_like(sdf, dtype=np.float64),
            np.zeros_like(sdf, dtype=np.float64),
        ),
    )
    runtime = SimpleNamespace(
        spatial_dim=2,
        geometry_provider=geometry_provider,
        field_provider=None,
    )
    reference = pd.DataFrame(
        [
            {'particle_id': 1, 'x': 0.6, 'y': 0.5, 'v_x': 1.0, 'v_y': 0.0, 'active': 1, 'stuck': 0, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 0},
            {'particle_id': 2, 'x': 0.4, 'y': 0.5, 'v_x': 0.0, 'v_y': 0.0, 'active': 0, 'stuck': 1, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 0},
        ]
    )
    candidate = pd.DataFrame(
        [
            {'particle_id': 1, 'x': 0.7, 'y': 0.5, 'v_x': 1.5, 'v_y': 0.0, 'active': 1, 'stuck': 0, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 0},
            {'particle_id': 2, 'x': 0.45, 'y': 0.5, 'v_x': 0.0, 'v_y': 0.0, 'active': 1, 'stuck': 0, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 0},
        ]
    )

    summary = geometry_feature_delta_summary(candidate, reference, runtime)

    assert summary['compared_particles'] == 2
    assert summary['position_error_m']['max'] == pytest.approx(0.1, abs=1.0e-12)
    assert summary['sdf_error_m']['max'] == pytest.approx(0.1, abs=1.0e-12)
    assert summary['nearest_boundary_distance_error_m']['count'] == 2
    assert summary['outside_geometry_count_candidate'] == 1
    assert summary['outside_geometry_count_delta'] == 0
    assert summary['mismatched_state_feature_summary']['count'] == 1


def test_state_contract_classification_matches_invalid_mask_priority():
    final_df = pd.DataFrame(
        [
            {'particle_id': 1, 'active': 1, 'stuck': 0, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 0},
            {'particle_id': 2, 'active': 1, 'stuck': 0, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 1},
            {'particle_id': 3, 'active': 1, 'stuck': 1, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 1},
            {'particle_id': 4, 'active': 0, 'stuck': 0, 'absorbed': 1, 'escaped': 1, 'invalid_mask_stopped': 0},
        ]
    )
    labels = classify_particle_states(final_df)
    classes = particle_class_frame(final_df)
    assert labels.tolist() == ['active_free_flight', 'invalid_mask_stopped', 'stuck', 'escaped']
    assert classes['particle_class'].tolist() == ['active_free_flight', 'invalid_mask_stopped', 'stuck', 'escaped']


def test_compare_against_reference_cli_writes_summary(tmp_path: Path):
    output_root = tmp_path / 'compare_runs'
    rc = compare_against_reference_main(
        [
            '--reference-config',
            str(ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml'),
            '--run',
            f"same={ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml'}",
            '--output-root',
            str(output_root),
        ]
    )
    assert rc == 0
    summary_files = sorted(output_root.glob('compare_*/comparison_summary.json'))
    assert summary_files
    summary = json.loads(summary_files[-1].read_text(encoding='utf-8'))
    assert summary['runs'][0]['run'] == 'same'
    assert summary['runs'][0]['class_match_ratio_vs_reference'] == pytest.approx(1.0)
    assert summary['runs'][0]['class_mismatch_count_vs_reference'] == 0
    assert summary['runs'][0]['class_transition_summary_vs_reference']['compared_particles'] > 0
    assert summary['runs'][0]['geometry_feature_delta_vs_reference']['compared_particles'] > 0
    assert summary['runs'][0]['geometry_feature_delta_vs_reference']['position_error_m']['max'] == pytest.approx(0.0, abs=1.0e-15)
    assert summary['runs'][0]['source_initial_geometry_summary']['particle_count'] > 0
    assert summary['runs'][0]['unresolved_crossing_count'] >= 0
    assert summary['runs'][0]['numerical_boundary_stopped_count'] == 0
    assert summary['runs'][0]['nearest_projection_fallback_count'] == 0
    assert summary['runs'][0]['boundary_event_failure_count'] == 0
    assert summary['runs'][0]['boundary_event_contract_passed'] == 1


def test_initial_particle_field_support_contract_rejects_non_clean_start(tmp_path: Path):
    axes = np.linspace(-1.0, 1.0, 81)
    field_path = tmp_path / 'field_initial_contract.npz'
    _write_field_bundle(field_path, axes, axes)
    payload = {key: value for key, value in np.load(field_path).items()}
    valid_mask = np.ones((axes.size, axes.size), dtype=bool)
    valid_mask[axes <= -0.75, :] = False
    payload['valid_mask'] = valid_mask
    np.savez_compressed(field_path, **payload)

    particles_path = tmp_path / 'particles_initial_contract.csv'
    pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': -0.8,
                'y': -0.2,
                'vx': 0.0,
                'vy': 0.0,
                'release_time': 0.25,
                'mass': 1e-15,
                'diameter': 1e-6,
                'density': 1200,
                'charge': 0,
                'source_part_id': 10,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    ).to_csv(particles_path, index=False)

    cfg_dir = tmp_path / 'cfg_initial_contract'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = _write_config(
        cfg_dir,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: (
            cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())}),
            cfg.setdefault('providers', {}).update(
                {'field': {'kind': 'precomputed_npz', 'npz_path': str(field_path.resolve())}}
            ),
            cfg.setdefault('provider_contract', {}).update({'boundary_field_support': 'off'}),
        ),
    )
    prepared = build_prepared_runtime_from_yaml(cfg)
    report = build_initial_particle_field_support_report(prepared)
    assert report['status_counts']['hard_invalid'] == 1
    assert report['status_counts']['non_clean'] == 1
    assert report['checked_time_min_s'] == pytest.approx(0.25)
    assert report['checked_time_max_s'] == pytest.approx(0.25)
    assert report['violations'][0]['checked_time_s'] == pytest.approx(0.25)

    out_dir = tmp_path / 'out_initial_contract'
    with pytest.raises(ValueError, match='Initial particles must be inside the clean field sample domain'):
        run_solver_2d_from_yaml(cfg, output_dir=out_dir)
    written = json.loads((out_dir / 'input_contract_report.json').read_text(encoding='utf-8'))
    assert written['status_counts']['hard_invalid'] == 1
    assert (out_dir / 'input_particle_violations.csv').exists()


def test_check_input_cli_exit_code_follows_contract_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    axes = np.linspace(-1.0, 1.0, 81)
    field_path = tmp_path / 'field_check_input_mode.npz'
    _write_field_bundle(field_path, axes, axes)
    payload = {key: value for key, value in np.load(field_path).items()}
    valid_mask = np.ones((axes.size, axes.size), dtype=bool)
    valid_mask[axes <= -0.75, :] = False
    payload['valid_mask'] = valid_mask
    np.savez_compressed(field_path, **payload)

    particles_path = tmp_path / 'particles_check_input_mode.csv'
    pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': -0.8,
                'y': -0.2,
                'vx': 0.0,
                'vy': 0.0,
                'release_time': 0.0,
                'mass': 1e-15,
                'diameter': 1e-6,
                'density': 1200,
                'charge': 0,
                'source_part_id': 10,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    ).to_csv(particles_path, index=False)

    def _config_for(mode: str, cfg_dir: Path) -> Path:
        cfg_dir.mkdir(parents=True, exist_ok=True)
        return _write_config(
            cfg_dir,
            ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
            mutate=lambda cfg: (
                cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())}),
                cfg.setdefault('providers', {}).update(
                    {'field': {'kind': 'precomputed_npz', 'npz_path': str(field_path.resolve())}}
                ),
                cfg.setdefault('input_contract', {}).update({'initial_particle_field_support': mode}),
                cfg.setdefault('provider_contract', {}).update({'boundary_field_support': 'off'}),
            ),
        )

    warn_cfg = _config_for('warn', tmp_path / 'cfg_warn')
    strict_cfg = _config_for('strict', tmp_path / 'cfg_strict')

    monkeypatch.setattr(
        sys,
        'argv',
        ['run_from_yaml.py', str(warn_cfg), '--check-input', '--output-dir', str(tmp_path / 'out_warn')],
    )
    assert run_from_yaml_main() == 0
    warn_report = json.loads((tmp_path / 'out_warn' / 'input_contract_report.json').read_text(encoding='utf-8'))
    assert warn_report['passed'] is True
    assert int(warn_report['status_counts']['non_clean']) == 1

    monkeypatch.setattr(
        sys,
        'argv',
        ['run_from_yaml.py', str(strict_cfg), '--check-input', '--output-dir', str(tmp_path / 'out_strict')],
    )
    assert run_from_yaml_main() == 1
    strict_report = json.loads((tmp_path / 'out_strict' / 'input_contract_report.json').read_text(encoding='utf-8'))
    assert strict_report['passed'] is False
    assert int(strict_report['status_counts']['non_clean']) == 1


def test_boundary_field_support_contract_rejects_boundary_adjacent_field_gap(tmp_path: Path):
    axes = np.linspace(-1.0, 1.0, 81)
    field_path = tmp_path / 'field_boundary_contract.npz'
    _write_field_bundle(field_path, axes, axes)
    payload = {key: value for key, value in np.load(field_path).items()}
    valid_mask = np.ones((axes.size, axes.size), dtype=bool)
    valid_mask[axes <= -0.95, :] = False
    payload['valid_mask'] = valid_mask
    np.savez_compressed(field_path, **payload)

    cfg = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: (
            cfg.setdefault('providers', {}).update(
                {'field': {'kind': 'precomputed_npz', 'npz_path': str(field_path.resolve())}}
            ),
            cfg.setdefault('provider_contract', {}).update(
                {
                    'boundary_field_support': 'strict',
                    'boundary_offset_cells': 0.5,
                    'max_boundary_samples': 400,
                }
            ),
            cfg.setdefault('source', {}).setdefault('preprocess', {}).update({'enabled': False}),
        ),
    )
    prepared = build_prepared_runtime_from_yaml(cfg)
    report = build_boundary_field_support_report(prepared)
    assert report['applicable'] is True
    assert report['passed'] is False
    assert report['status_counts']['non_clean'] > 0
    assert report['violation_count'] == report['status_counts']['non_clean']
    assert report['violations_truncated'] is False
    assert int(report['field_support']['valid_node_count']) < int(report['field_support']['grid_node_count'])
    summaries = {int(row['part_id']): row for row in report['violation_summary_by_part']}
    assert summaries
    assert sum(int(row['violation_count']) for row in summaries.values()) == report['status_counts']['non_clean']
    assert all(len(row['boundary_min']) == 2 for row in summaries.values())

    out_dir = tmp_path / 'out_boundary_contract'
    with pytest.raises(ValueError, match='Field provider does not cover the explicit boundary support domain'):
        run_solver_2d_from_yaml(cfg, output_dir=out_dir)
    written = json.loads((out_dir / 'provider_contract_report.json').read_text(encoding='utf-8'))
    assert written['status_counts']['non_clean'] > 0
    assert written['violation_count'] == written['status_counts']['non_clean']
    assert sum(int(row['violation_count']) for row in written['violation_summary_by_part']) == written['status_counts']['non_clean']
    assert (out_dir / 'provider_boundary_summary.csv').exists()
    summary_csv = pd.read_csv(out_dir / 'provider_boundary_summary.csv')
    assert len(summary_csv) == len(written['violation_summary_by_part'])
    assert int(summary_csv['violation_count'].sum()) == written['status_counts']['non_clean']
    assert {'boundary_min_x', 'boundary_max_y', 'offset_min_x', 'offset_max_y'}.issubset(set(summary_csv.columns))
    assert (out_dir / 'provider_boundary_violations.csv').exists()
    violations_csv = pd.read_csv(out_dir / 'provider_boundary_violations.csv')
    assert len(violations_csv) == written['status_counts']['non_clean']
    assert set(violations_csv.columns) == {
        'sample_index',
        'part_id',
        'boundary_index',
        'sample_kind',
        'checked_time_s',
        'status',
        'boundary_x',
        'boundary_y',
        'offset_x',
        'offset_y',
    }


def test_boundary_field_support_contract_reports_transient_checked_times(tmp_path: Path):
    axes = np.linspace(-1.0, 1.0, 81)
    shape = (axes.size, axes.size)
    xx, yy = np.meshgrid(axes, axes, indexing='ij')
    times = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    valid_mask = np.ones(shape, dtype=bool)
    valid_mask[axes <= -0.95, :] = False
    field_path = tmp_path / 'field_boundary_transient_contract.npz'
    np.savez_compressed(
        field_path,
        axis_0=axes,
        axis_1=axes,
        times=times,
        valid_mask=valid_mask,
        ux=np.stack([(1.0 + t) * np.ones(shape, dtype=np.float64) for t in times], axis=0),
        uy=np.stack([0.05 * np.cos(xx * 10.0) + 0.0 * yy for _ in times], axis=0),
        mu=np.stack([1.8e-5 * np.ones(shape, dtype=np.float64) for _ in times], axis=0),
    )

    cfg = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: (
            cfg.setdefault('providers', {}).update(
                {'field': {'kind': 'precomputed_npz', 'npz_path': str(field_path.resolve())}}
            ),
            cfg.setdefault('provider_contract', {}).update(
                {
                    'boundary_field_support': 'strict',
                    'boundary_offset_cells': 0.5,
                    'max_boundary_samples': 100,
                    'max_time_samples': 2,
                }
            ),
            cfg.setdefault('source', {}).setdefault('preprocess', {}).update({'enabled': False}),
        ),
    )
    prepared = build_prepared_runtime_from_yaml(cfg)
    report = build_boundary_field_support_report(prepared)

    assert report['field_support']['field_backend_kind'] == 'regular_rectilinear'
    assert report['field_support']['time_axis']['time_count'] == 3
    assert report['field_support']['time_axis']['quantity_time_axis_mismatch_count'] == 0
    assert report['checked_times_s'] == [0.0, 1.0]
    assert int(report['checked_time_count']) == 2
    assert int(report['sample_count']) == int(report['boundary_point_sample_count']) * 2
    assert report['status_counts']['non_clean'] > 0
    assert all('checked_time_s' in row for row in report['violations'])


def test_3d_provider_contract_samples_face_edge_and_vertex_neighborhoods():
    prepared = build_prepared_runtime_from_yaml(ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml')
    report = build_boundary_field_support_report(prepared)
    kind_counts = report['boundary_sample_kind_counts']
    geometry_boundary = report['geometry_boundary']

    assert report['passed'] is True
    assert geometry_boundary['available'] is True
    assert geometry_boundary['boundary_triangle_count'] == 12
    assert geometry_boundary['boundary_surface_validation']['triangle_count'] == 12
    assert kind_counts['face_centroid'] > 0
    assert kind_counts['edge_mid_0'] > 0
    assert kind_counts['vertex_0'] > 0
    assert int(report['boundary_point_sample_count']) > int(kind_counts['face_centroid'])


def test_integrator_specs_expose_expected_internal_capabilities():
    drag = get_integrator_spec('drag_relaxation')
    etd = get_integrator_spec('etd')
    etd2 = get_integrator_spec('etd2')
    assert drag.mode == 0
    assert drag.order == 1
    assert drag.uses_midpoint_stage is False
    assert drag.stage_point_count == 1
    assert etd.mode == 1
    assert etd.supports_partial_replay is True
    assert etd2.mode == 2
    assert etd2.order == 2
    assert etd2.uses_midpoint_stage is True
    assert etd2.stage_point_count == 2
    assert integrator_spec_from_mode(2) == etd2


def test_unsupported_integrator_raises(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('solver', {}).update({'integrator': 'euler'}),
    )
    with pytest.raises(ValueError, match='Unsupported solver.integrator'):
        build_prepared_runtime_from_yaml(config_path)


def test_eit_integrator_alias_is_rejected(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('solver', {}).update({'integrator': 'eit'}),
    )
    with pytest.raises(ValueError, match='Unsupported solver.integrator'):
        build_prepared_runtime_from_yaml(config_path)


def test_contact_tangent_motion_config_is_rejected(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('solver', {}).update({'contact_tangent_motion': True}),
    )

    with pytest.raises(ValueError, match='contact_tangent_motion is obsolete'):
        run_solver_2d_from_yaml(config_path, output_dir=tmp_path / 'out_contact_tangent_rejected')


def test_etd_integrator_runs_in_2d(tmp_path: Path):
    out_dir = tmp_path / 'out_2d_etd'
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('solver', {}).update({'integrator': 'etd', 't_end': 0.1, 'save_every': 1}),
    )
    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    report = json.loads((out_dir / 'solver_report.json').read_text(encoding='utf-8'))
    diag = json.loads((out_dir / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    assert report['integrator'] == 'etd'
    assert diag['integrator'] == 'etd'


def test_etd2_integrator_runs_in_2d_and_3d(tmp_path: Path):
    out_2d = tmp_path / 'out_2d_etd2'
    cfg_2d_dir = tmp_path / 'cfg_2d_etd2'
    cfg_2d_dir.mkdir(parents=True, exist_ok=True)
    cfg_2d = _write_config(
        cfg_2d_dir,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('solver', {}).update({'integrator': 'etd2', 't_end': 0.1, 'save_every': 1}),
    )
    run_solver_2d_from_yaml(cfg_2d, output_dir=out_2d)
    report_2d = json.loads((out_2d / 'solver_report.json').read_text(encoding='utf-8'))
    diag_2d = json.loads((out_2d / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    assert report_2d['integrator'] == 'etd2'
    assert diag_2d['integrator'] == 'etd2'
    assert report_2d['boundary_event_contract_passed'] == 1
    assert report_2d['boundary_event_failure_count'] == 0
    assert diag_2d['boundary_event_contract_passed'] == 1
    assert set(diag_2d).issuperset(
        {
            'etd2_polyline_checks_count',
            'etd2_midpoint_outside_count',
            'etd2_polyline_hit_count',
            'etd2_polyline_fallback_count',
            'state_geometry_summary',
            'source_initial_geometry_summary',
        }
    )
    assert int(diag_2d['state_geometry_summary']['particle_count']) > 0
    assert 'active' in diag_2d['state_geometry_summary']['by_state']
    assert int(diag_2d['source_initial_geometry_summary']['particle_count']) > 0
    assert 'released_by_end' in diag_2d['source_initial_geometry_summary']['by_release_state']

    out_3d = tmp_path / 'out_3d_etd2'
    cfg_3d_dir = tmp_path / 'cfg_3d_etd2'
    cfg_3d_dir.mkdir(parents=True, exist_ok=True)
    cfg_3d = _write_config(
        cfg_3d_dir,
        ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('solver', {}).update({'integrator': 'etd2', 't_end': 0.1, 'save_every': 1}),
    )
    run_solver_3d_from_yaml(cfg_3d, output_dir=out_3d)
    report_3d = json.loads((out_3d / 'solver_report.json').read_text(encoding='utf-8'))
    diag_3d = json.loads((out_3d / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    assert report_3d['integrator'] == 'etd2'
    assert diag_3d['integrator'] == 'etd2'
    assert report_3d['boundary_event_contract_passed'] == 1
    assert report_3d['boundary_event_failure_count'] == 0
    assert diag_3d['boundary_event_contract_passed'] == 1
    assert set(diag_3d).issuperset(
        {
            'etd2_polyline_checks_count',
            'etd2_midpoint_outside_count',
            'etd2_polyline_hit_count',
            'etd2_polyline_fallback_count',
        }
    )


def _final_xy_velocity(out_dir: Path) -> np.ndarray:
    df = pd.read_csv(out_dir / 'final_particles.csv')
    return df[['x', 'y', 'v_x', 'v_y']].to_numpy(dtype=np.float64)


def test_stochastic_motion_disabled_preserves_solver_outputs(tmp_path: Path):
    for name in ('cfg_base', 'cfg_disabled'):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)

    def _base_mutation(cfg):
        cfg.setdefault('solver', {}).update({'t_end': 0.06, 'save_every': 1, 'seed': 2468})
        cfg.setdefault('output', {}).update({'artifact_mode': 'minimal'})

    cfg_base = _write_config(
        tmp_path / 'cfg_base',
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=_base_mutation,
    )

    def _disabled_mutation(cfg):
        _base_mutation(cfg)
        cfg.setdefault('solver', {})['stochastic_motion'] = {'enabled': False}

    cfg_disabled = _write_config(
        tmp_path / 'cfg_disabled',
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=_disabled_mutation,
    )
    out_base = tmp_path / 'out_base'
    out_disabled = tmp_path / 'out_disabled'
    run_solver_2d_from_yaml(cfg_base, output_dir=out_base)
    run_solver_2d_from_yaml(cfg_disabled, output_dir=out_disabled)
    np.testing.assert_allclose(_final_xy_velocity(out_base), _final_xy_velocity(out_disabled), rtol=0.0, atol=0.0)
    report = json.loads((out_disabled / 'solver_report.json').read_text(encoding='utf-8'))
    assert report['stochastic_motion']['enabled'] == 0


def test_stochastic_motion_seed_is_reproducible_and_changes_trajectory(tmp_path: Path):
    for name in ('cfg_a', 'cfg_b', 'cfg_c'):
        (tmp_path / name).mkdir(parents=True, exist_ok=True)

    def _mutate(seed: int):
        def _inner(cfg):
            cfg.setdefault('solver', {}).update(
                {
                    't_end': 0.06,
                    'save_every': 1,
                    'seed': 1234,
                    'stochastic_motion': {
                        'enabled': True,
                        'model': 'underdamped_langevin',
                        'stride': 1,
                        'seed': int(seed),
                        'temperature_source': 'gas',
                    },
                }
            )
            cfg.setdefault('output', {}).update({'artifact_mode': 'minimal'})

        return _inner

    cfg_a = _write_config(tmp_path / 'cfg_a', ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml', mutate=_mutate(77))
    cfg_b = _write_config(tmp_path / 'cfg_b', ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml', mutate=_mutate(77))
    cfg_c = _write_config(tmp_path / 'cfg_c', ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml', mutate=_mutate(78))
    out_a = tmp_path / 'out_a'
    out_b = tmp_path / 'out_b'
    out_c = tmp_path / 'out_c'
    run_solver_2d_from_yaml(cfg_a, output_dir=out_a)
    run_solver_2d_from_yaml(cfg_b, output_dir=out_b)
    run_solver_2d_from_yaml(cfg_c, output_dir=out_c)

    np.testing.assert_allclose(_final_xy_velocity(out_a), _final_xy_velocity(out_b), rtol=0.0, atol=0.0)
    assert not np.allclose(_final_xy_velocity(out_a), _final_xy_velocity(out_c), rtol=0.0, atol=1.0e-12)
    report = json.loads((out_a / 'solver_report.json').read_text(encoding='utf-8'))
    stochastic = report['stochastic_motion']
    assert stochastic['enabled'] == 1
    assert stochastic['model'] == 'underdamped_langevin'
    assert stochastic['kick_event_count'] > 0
    assert stochastic['kicked_particle_count'] > 0
    assert stochastic['velocity_kick_rms_mps'] > 0.0


def test_valid_mask_diagnostics_do_not_change_solver_outputs(tmp_path: Path):
    axes = np.linspace(-1.0, 1.0, 81)
    base_field_path = tmp_path / 'field_all_true.npz'
    masked_field_path = tmp_path / 'field_masked.npz'
    _write_field_bundle(base_field_path, axes, axes)
    payload = {key: value for key, value in np.load(base_field_path).items()}
    valid_mask = np.ones((axes.size, axes.size), dtype=bool)
    valid_mask[axes <= -0.75, :] = False
    payload['valid_mask'] = valid_mask
    np.savez_compressed(masked_field_path, **payload)

    def _field_override(npz_path: Path):
        def mutate(cfg):
            cfg.setdefault('providers', {})['field'] = {
                'kind': 'precomputed_npz',
                'npz_path': str(npz_path.resolve()),
            }
            cfg.setdefault('solver', {}).update(
                {'integrator': 'etd2', 't_end': 0.12, 'save_every': 1, 'valid_mask_policy': 'diagnostic'}
            )
            cfg.setdefault('input_contract', {}).update({'initial_particle_field_support': 'warn'})
            cfg.setdefault('provider_contract', {}).update({'boundary_field_support': 'off'})
        return mutate

    cfg_base_dir = tmp_path / 'cfg_base'
    cfg_masked_dir = tmp_path / 'cfg_masked'
    cfg_base_dir.mkdir(parents=True, exist_ok=True)
    cfg_masked_dir.mkdir(parents=True, exist_ok=True)
    cfg_base = _write_config(cfg_base_dir, ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml', mutate=_field_override(base_field_path))
    cfg_masked = _write_config(cfg_masked_dir, ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml', mutate=_field_override(masked_field_path))

    out_base = tmp_path / 'out_base'
    out_masked = tmp_path / 'out_masked'
    run_solver_2d_from_yaml(cfg_base, output_dir=out_base)
    run_solver_2d_from_yaml(cfg_masked, output_dir=out_masked)

    base_final = pd.read_csv(out_base / 'final_particles.csv').sort_values('particle_id').reset_index(drop=True)
    masked_final = pd.read_csv(out_masked / 'final_particles.csv').sort_values('particle_id').reset_index(drop=True)
    assert base_final[['particle_id', 'active', 'stuck', 'absorbed', 'escaped']].equals(
        masked_final[['particle_id', 'active', 'stuck', 'absorbed', 'escaped']]
    )
    assert 'invalid_mask_stopped' in masked_final.columns
    assert int(masked_final['invalid_mask_stopped'].sum()) == 0
    for col in ('x', 'y', 'v_x', 'v_y'):
        assert masked_final[col].to_numpy() == pytest.approx(base_final[col].to_numpy(), abs=1.0e-12)

    base_report = json.loads((out_base / 'solver_report.json').read_text(encoding='utf-8'))
    masked_report = json.loads((out_masked / 'solver_report.json').read_text(encoding='utf-8'))
    masked_steps = pd.read_csv(out_masked / 'runtime_step_summary.csv')
    assert str(base_report['valid_mask_policy']) == 'diagnostic'
    assert str(masked_report['valid_mask_policy']) == 'diagnostic'
    assert int(base_report['valid_mask_violation_count']) == 0
    assert int(masked_report['valid_mask_violation_count']) > 0
    assert int(masked_report['valid_mask_violation_particle_count']) > 0
    assert int(masked_report['valid_mask_violation_count']) == int(masked_report['valid_mask_mixed_stencil_count']) + int(
        masked_report['valid_mask_hard_invalid_count']
    )
    assert int(masked_report['invalid_mask_stopped_count']) == 0
    assert 'valid_mask_violation_count_step' in masked_steps.columns
    assert 'valid_mask_mixed_stencil_count_step' in masked_steps.columns
    assert 'valid_mask_hard_invalid_count_step' in masked_steps.columns
    assert 'invalid_mask_stopped_count_step' in masked_steps.columns
    assert int(masked_steps['valid_mask_violation_count_step'].sum()) > 0
    assert int(masked_steps['valid_mask_violation_count_step'].sum()) == int(
        masked_steps['valid_mask_mixed_stencil_count_step'].sum() + masked_steps['valid_mask_hard_invalid_count_step'].sum()
    )
    assert int(masked_steps['invalid_mask_stopped_count_step'].sum()) == 0


def test_valid_mask_retry_then_stop_does_not_stop_on_mixed_stencil_only():
    state = SimpleNamespace(
        active=np.asarray([True], dtype=bool),
        valid_mask_status_flags=np.asarray([VALID_MASK_STATUS_MIXED_STENCIL], dtype=np.uint8),
        x=np.asarray([[-0.6, -0.2]], dtype=np.float64),
        v=np.asarray([[0.0, 0.0]], dtype=np.float64),
        x_trial=np.asarray([[-0.52, -0.2]], dtype=np.float64),
        v_trial=np.asarray([[1.0, 0.0]], dtype=np.float64),
        x_mid_trial=np.asarray([[-0.56, -0.2]], dtype=np.float64),
        invalid_mask_stopped=np.asarray([False], dtype=bool),
        invalid_stop_reason_code=np.asarray([0], dtype=np.uint8),
        stuck=np.asarray([False], dtype=bool),
        absorbed=np.asarray([False], dtype=bool),
        escaped=np.asarray([False], dtype=bool),
        collision_diagnostics=_collision_diag_stub(),
    )
    options = SimpleNamespace(
        valid_mask_policy='retry_then_stop',
        adaptive_substep_enabled=0,
        adaptive_substep_tau_ratio=0.5,
        adaptive_substep_max_splits=4,
        drag_model_mode=0,
    )

    stopped_count = _apply_valid_mask_retry_then_stop(
        state=state,
        options=options,
        compiled={},
        spatial_dim=2,
        integrator_mode=0,
        dt_step=0.08,
        t_end_step=0.08,
        phys={
            'flow_scale': 1.0,
            'drag_tau_scale': 1.0,
            'body_accel_scale': 1.0,
            'min_tau_p_s': 1.0e-9,
            'gas_density_kgm3': 1.2,
            'gas_mu_pas': 1.8e-5,
        },
        body_accel=np.zeros(2, dtype=np.float64),
        tau_p=np.asarray([1.0], dtype=np.float64),
        particle_diameter=np.asarray([1.0e-6], dtype=np.float64),
        flow_scale_particle=np.asarray([1.0], dtype=np.float64),
        drag_scale_particle=np.asarray([1.0], dtype=np.float64),
        body_scale_particle=np.asarray([1.0], dtype=np.float64),
    )

    assert int(stopped_count) == 0
    assert bool(state.invalid_mask_stopped[0]) is False
    assert int(state.collision_diagnostics['invalid_mask_retry_count']) == 0
    assert int(state.collision_diagnostics['invalid_mask_stopped_count']) == 0


def test_valid_mask_retry_then_stop_stops_particle_at_last_valid_prefix(tmp_path: Path):
    axes = np.linspace(-1.0, 1.0, 81)
    field_path = tmp_path / 'field_retry_then_stop.npz'
    _write_field_bundle(field_path, axes, axes)
    payload = {key: value for key, value in np.load(field_path).items()}
    payload['ux'] = 4.0 * np.ones((axes.size, axes.size), dtype=np.float64)
    payload['uy'] = np.zeros((axes.size, axes.size), dtype=np.float64)
    valid_mask = np.ones((axes.size, axes.size), dtype=bool)
    valid_mask[axes >= -0.5, :] = False
    payload['valid_mask'] = valid_mask
    np.savez_compressed(field_path, **payload)

    particles_path = tmp_path / 'particles_single.csv'
    pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': -0.8,
                'y': -0.2,
                'vx': 0.2,
                'vy': 0.0,
                'release_time': 0.0,
                'mass': 1e-15,
                'diameter': 1e-6,
                'density': 1200,
                'charge': 0,
                'source_part_id': 10,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    ).to_csv(particles_path, index=False)

    cfg_dir = tmp_path / 'cfg_retry_then_stop'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = _write_config(
        cfg_dir,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: (
            cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())}),
            cfg.setdefault('providers', {}).update(
                {
                    'field': {
                        'kind': 'precomputed_npz',
                        'npz_path': str(field_path.resolve()),
                    }
                }
            ),
            cfg.setdefault('solver', {}).update(
                    {
                        't_end': 0.2,
                        'dt': 0.2,
                        'save_every': 1,
                        'integrator': 'etd2',
                        'adaptive_substep_max_splits': 4,
                        }
                    ),
                cfg.setdefault('provider_contract', {}).update({'boundary_field_support': 'off'}),
                ),
    )

    out_dir = tmp_path / 'out_retry_then_stop'
    run_solver_2d_from_yaml(cfg, output_dir=out_dir)

    final_df = pd.read_csv(out_dir / 'final_particles.csv')
    report = json.loads((out_dir / 'solver_report.json').read_text(encoding='utf-8'))
    diag = json.loads((out_dir / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    step_df = pd.read_csv(out_dir / 'runtime_step_summary.csv')
    wall_df = pd.read_csv(out_dir / 'wall_events.csv')

    row = final_df.loc[0]
    assert int(row['invalid_mask_stopped']) == 1
    assert int(row['active']) == 0
    assert int(row['stuck']) == 0
    assert int(row['absorbed']) == 0
    assert int(row['escaped']) == 0
    assert float(row['x']) > -0.8
    assert float(row['x']) < 0.0
    assert str(report['valid_mask_policy']) == 'retry_then_stop'
    assert int(report['invalid_mask_stopped_count']) == 1
    assert int(diag['invalid_mask_stopped_count']) == 1
    assert int(diag['invalid_mask_retry_count']) > 0
    assert int(diag['invalid_mask_retry_exhausted_count']) == 0
    assert str(row['invalid_stop_reason']) == 'freeflight_valid_mask_hard_invalid_prefix_clipped'
    assert int(diag['invalid_mask_stop_reason_counts']['freeflight_valid_mask_hard_invalid_prefix_clipped']) == 1
    assert int(report['invalid_stop_geometry_summary']['count']) == 1
    assert int(diag['invalid_stop_geometry_summary']['count']) == 1
    assert int(diag['invalid_stop_geometry_summary']['sdf_m']['count']) == 1
    assert int(diag['invalid_stop_geometry_summary']['nearest_boundary_distance_m']['count']) == 1
    assert diag['invalid_stop_geometry_summary']['nearest_part_counts']
    assert int(report['state_geometry_summary']['by_state']['invalid_mask_stopped']['count']) == 1
    assert int(diag['state_geometry_summary']['by_state']['invalid_mask_stopped']['sdf_m']['count']) == 1
    assert int(step_df['invalid_mask_stopped_count_step'].sum()) == 1
    assert wall_df.empty or int((wall_df['particle_id'] == 1).sum()) == 0


def test_valid_mask_retry_then_stop_keeps_particle_at_pre_step_when_no_valid_prefix_exists(tmp_path: Path):
    axes = np.linspace(-1.0, 1.0, 81)
    field_path = tmp_path / 'field_retry_exhausted.npz'
    _write_field_bundle(field_path, axes, axes)
    payload = {key: value for key, value in np.load(field_path).items()}
    payload['ux'] = 2.0 * np.ones((axes.size, axes.size), dtype=np.float64)
    payload['uy'] = np.zeros((axes.size, axes.size), dtype=np.float64)
    valid_mask = np.ones((axes.size, axes.size), dtype=bool)
    valid_mask[axes <= -0.75, :] = False
    payload['valid_mask'] = valid_mask
    np.savez_compressed(field_path, **payload)

    particles_path = tmp_path / 'particles_single_invalid_start.csv'
    pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': -0.8,
                'y': -0.2,
                'vx': 0.2,
                'vy': 0.0,
                'release_time': 0.0,
                'mass': 1e-15,
                'diameter': 1e-6,
                'density': 1200,
                'charge': 0,
                'source_part_id': 10,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    ).to_csv(particles_path, index=False)

    cfg_dir = tmp_path / 'cfg_retry_exhausted'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = _write_config(
        cfg_dir,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: (
            cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())}),
            cfg.setdefault('providers', {}).update(
                {
                    'field': {
                        'kind': 'precomputed_npz',
                        'npz_path': str(field_path.resolve()),
                    }
                }
            ),
            cfg.setdefault('solver', {}).update(
                {
                    't_end': 0.12,
                    'dt': 0.12,
                    'save_every': 1,
                    'adaptive_substep_max_splits': 4,
                    'valid_mask_policy': 'retry_then_stop',
                }
            ),
            cfg.setdefault('input_contract', {}).update({'initial_particle_field_support': 'warn'}),
            cfg.setdefault('provider_contract', {}).update({'boundary_field_support': 'off'}),
        ),
    )

    out_dir = tmp_path / 'out_retry_exhausted'
    run_solver_2d_from_yaml(cfg, output_dir=out_dir)

    final_df = pd.read_csv(out_dir / 'final_particles.csv')
    report = json.loads((out_dir / 'solver_report.json').read_text(encoding='utf-8'))
    diag = json.loads((out_dir / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    row = final_df.loc[0]
    assert int(row['invalid_mask_stopped']) == 1
    assert int(row['active']) == 0
    assert int(row['stuck']) == 0
    assert int(row['absorbed']) == 0
    assert int(row['escaped']) == 0
    assert float(row['x']) == pytest.approx(-0.8, abs=1.0e-12)
    assert float(row['y']) == pytest.approx(-0.2, abs=1.0e-12)
    assert float(row['v_x']) == pytest.approx(0.2, abs=1.0e-12)
    assert float(row['v_y']) == pytest.approx(0.0, abs=1.0e-12)
    assert int(report['invalid_mask_stopped_count']) == 1
    assert int(diag['invalid_mask_stopped_count']) == 1
    assert int(diag['invalid_mask_retry_exhausted_count']) == 1
    assert str(row['invalid_stop_reason']) == 'freeflight_valid_mask_hard_invalid_retry_exhausted'
    assert int(report['invalid_mask_stop_reason_counts']['freeflight_valid_mask_hard_invalid_retry_exhausted']) == 1
    assert int(report['invalid_stop_geometry_summary']['count']) == 1
    assert int(report['state_geometry_summary']['by_state']['invalid_mask_stopped']['count']) == 1


def test_collision_replay_retry_then_stop_marks_invalid_stop_before_extra_wall_events(monkeypatch: pytest.MonkeyPatch):
    diag = _collision_diag_stub()
    wall_rows: list[dict[str, object]] = []
    max_hit_rows: list[dict[str, object]] = []
    wall_law_counts: dict[str, int] = {}
    wall_summary_counts: dict[tuple[int, str, str], int] = {}
    stuck = np.zeros(1, dtype=bool)
    absorbed = np.zeros(1, dtype=bool)
    active = np.ones(1, dtype=bool)

    step = ProcessStepRow(step_id=1, step_name='run', start_s=0.0, end_s=1.0, output_segment_name='run')
    particles = SimpleNamespace(
        particle_id=np.asarray([1], dtype=np.int64),
        stick_probability=np.asarray([0.0], dtype=np.float64),
    )

    monkeypatch.setattr(
        'particle_tracer_unified.solvers.high_fidelity_collision.locate_physical_hit_state',
        lambda **kwargs: (
            BoundaryHit(
                position=np.asarray([-1.0, 0.0], dtype=np.float64),
                normal=np.asarray([-1.0, 0.0], dtype=np.float64),
                part_id=10,
                alpha_hint=0.1,
            ),
            np.asarray([-4.0, 0.0], dtype=np.float64),
            0.05,
        ),
    )
    monkeypatch.setattr(
        'particle_tracer_unified.solvers.high_fidelity_collision._apply_wall_hit_step',
        lambda **kwargs: (
            wall_rows.append(
                {
                    'particle_id': int(kwargs['particles'].particle_id[kwargs['particle_index']]),
                    'part_id': int(kwargs['part_id']),
                    'outcome': 'reflected_specular',
                }
            ),
            WallHitStepResult(
                np.asarray([-0.999, 0.0], dtype=np.float64),
                np.asarray([4.0, 0.0], dtype=np.float64),
                0.55,
                1,
                1,
                False,
            ),
        )[1],
    )
    monkeypatch.setattr(
        'particle_tracer_unified.solvers.high_fidelity_collision.advance_freeflight_segment',
        lambda **kwargs: (
            np.asarray([0.8, 0.0], dtype=np.float64),
            np.asarray([4.0, 0.0], dtype=np.float64),
            1,
            np.asarray([[0.2, 0.0]], dtype=np.float64),
            VALID_MASK_STATUS_HARD_INVALID,
        ),
    )
    monkeypatch.setattr(
        'particle_tracer_unified.solvers.high_fidelity_collision.resolve_valid_mask_prefix',
        lambda **kwargs: ValidMaskPrefixResolution(
            position=np.asarray([0.2, 0.0], dtype=np.float64),
            velocity=np.asarray([4.0, 0.0], dtype=np.float64),
            accepted_dt=0.275,
            retry_count=2,
            found_valid_prefix=True,
        ),
    )

    result = _advance_colliding_particle(
        runtime=SimpleNamespace(),
        step=step,
        particles=particles,
        particle_index=0,
        rng=np.random.default_rng(123),
        t=0.6,
        x_start=np.asarray([-0.8, 0.0], dtype=np.float64),
        v_start=np.asarray([-4.0, 0.0], dtype=np.float64),
        dt_step=0.6,
        spatial_dim=2,
        compiled={},
        integrator_mode=0,
        base_adaptive_substep_enabled=0,
        adaptive_substep_tau_ratio=0.5,
        adaptive_substep_max_splits=4,
        min_remaining_dt_ratio=0.0,
        tau_p_i=1.0,
        particle_diameter_i=1.0e-6,
        flow_scale_particle_i=1.0,
        drag_scale_particle_i=1.0,
        body_scale_particle_i=1.0,
        global_flow_scale=1.0,
        global_drag_tau_scale=1.0,
        global_body_accel_scale=1.0,
        body_accel=np.zeros(2, dtype=np.float64),
        min_tau_p_s=1.0e-9,
        gas_density_kgm3=1.2,
        gas_mu_pas=1.8e-5,
        drag_model_mode=0,
        valid_mask_retry_then_stop_enabled=True,
        initial_x_next=np.asarray([-3.2, 0.0], dtype=np.float64),
        initial_v_next=np.asarray([-4.0, 0.0], dtype=np.float64),
        initial_stage_points=np.asarray([[-3.2, 0.0]], dtype=np.float64),
        initial_valid_mask_status=VALID_MASK_STATUS_CLEAN,
        initial_primary_hit=BoundaryHit(
            position=np.asarray([-1.0, 0.0], dtype=np.float64),
            normal=np.asarray([-1.0, 0.0], dtype=np.float64),
            part_id=10,
            alpha_hint=0.1,
        ),
        initial_primary_hit_counted=False,
        inside_fn=lambda _p: True,
        strict_inside_fn=lambda _p: True,
        primary_hit_fn=lambda _p0, _stage: None,
        nearest_projection_fn=lambda _p, _anchor: None,
        primary_hit_counter_key='edge_hit_count',
        collision_diagnostics=diag,
        max_hit_rows=max_hit_rows,
        wall_rows=wall_rows,
        coating_summary_rows=[],
        wall_law_counts=wall_law_counts,
        wall_summary_counts=wall_summary_counts,
        stuck=stuck,
        absorbed=absorbed,
        active=active,
        max_wall_hits_per_step=5,
        epsilon_offset_m=1.0e-6,
        on_boundary_tol_m=1.0e-6,
        triangle_surface_3d=None,
    )

    assert bool(result.invalid_mask_stopped) is True
    assert str(result.invalid_stop_reason) == 'collision_valid_mask_hard_invalid_prefix_clipped'
    assert int(result.total_hits) == 1
    assert result.position == pytest.approx([0.2, 0.0])
    assert result.velocity == pytest.approx([4.0, 0.0])
    assert int(diag['invalid_mask_retry_count']) == 2
    assert int(diag['invalid_mask_retry_exhausted_count']) == 0
    assert int(diag['collision_reintegrated_segments_count']) == 1
    assert len(wall_rows) == 1
    assert len(max_hit_rows) == 0


def test_collision_replay_retry_then_stop_keeps_segment_start_when_retry_budget_exhausts(monkeypatch: pytest.MonkeyPatch):
    diag = _collision_diag_stub()
    wall_rows: list[dict[str, object]] = []

    step = ProcessStepRow(step_id=1, step_name='run', start_s=0.0, end_s=1.0, output_segment_name='run')
    particles = SimpleNamespace(
        particle_id=np.asarray([1], dtype=np.int64),
        stick_probability=np.asarray([0.0], dtype=np.float64),
    )

    monkeypatch.setattr(
        'particle_tracer_unified.solvers.high_fidelity_collision.locate_physical_hit_state',
        lambda **kwargs: (
            BoundaryHit(
                position=np.asarray([-1.0, 0.0], dtype=np.float64),
                normal=np.asarray([-1.0, 0.0], dtype=np.float64),
                part_id=10,
                alpha_hint=0.1,
            ),
            np.asarray([-4.0, 0.0], dtype=np.float64),
            0.05,
        ),
    )
    monkeypatch.setattr(
        'particle_tracer_unified.solvers.high_fidelity_collision._apply_wall_hit_step',
        lambda **kwargs: (
            wall_rows.append({'particle_id': 1, 'part_id': int(kwargs['part_id'])}),
            WallHitStepResult(
                np.asarray([-0.999, 0.0], dtype=np.float64),
                np.asarray([4.0, 0.0], dtype=np.float64),
                0.55,
                1,
                1,
                False,
            ),
        )[1],
    )
    monkeypatch.setattr(
        'particle_tracer_unified.solvers.high_fidelity_collision.advance_freeflight_segment',
        lambda **kwargs: (
            np.asarray([-0.7, 0.0], dtype=np.float64),
            np.asarray([4.0, 0.0], dtype=np.float64),
            1,
            np.asarray([[-0.7, 0.0]], dtype=np.float64),
            VALID_MASK_STATUS_HARD_INVALID,
        ),
    )
    monkeypatch.setattr(
        'particle_tracer_unified.solvers.high_fidelity_collision.resolve_valid_mask_prefix',
        lambda **kwargs: ValidMaskPrefixResolution(
            position=np.asarray([-0.999, 0.0], dtype=np.float64),
            velocity=np.asarray([4.0, 0.0], dtype=np.float64),
            accepted_dt=0.0,
            retry_count=4,
            found_valid_prefix=False,
        ),
    )

    result = _advance_colliding_particle(
        runtime=SimpleNamespace(),
        step=step,
        particles=particles,
        particle_index=0,
        rng=np.random.default_rng(123),
        t=0.6,
        x_start=np.asarray([-0.8, 0.0], dtype=np.float64),
        v_start=np.asarray([-4.0, 0.0], dtype=np.float64),
        dt_step=0.6,
        spatial_dim=2,
        compiled={},
        integrator_mode=0,
        base_adaptive_substep_enabled=0,
        adaptive_substep_tau_ratio=0.5,
        adaptive_substep_max_splits=4,
        min_remaining_dt_ratio=0.0,
        tau_p_i=1.0,
        particle_diameter_i=1.0e-6,
        flow_scale_particle_i=1.0,
        drag_scale_particle_i=1.0,
        body_scale_particle_i=1.0,
        global_flow_scale=1.0,
        global_drag_tau_scale=1.0,
        global_body_accel_scale=1.0,
        body_accel=np.zeros(2, dtype=np.float64),
        min_tau_p_s=1.0e-9,
        gas_density_kgm3=1.2,
        gas_mu_pas=1.8e-5,
        drag_model_mode=0,
        valid_mask_retry_then_stop_enabled=True,
        initial_x_next=np.asarray([-3.2, 0.0], dtype=np.float64),
        initial_v_next=np.asarray([-4.0, 0.0], dtype=np.float64),
        initial_stage_points=np.asarray([[-3.2, 0.0]], dtype=np.float64),
        initial_valid_mask_status=VALID_MASK_STATUS_CLEAN,
        initial_primary_hit=BoundaryHit(
            position=np.asarray([-1.0, 0.0], dtype=np.float64),
            normal=np.asarray([-1.0, 0.0], dtype=np.float64),
            part_id=10,
            alpha_hint=0.1,
        ),
        initial_primary_hit_counted=False,
        inside_fn=lambda _p: True,
        strict_inside_fn=lambda _p: True,
        primary_hit_fn=lambda _p0, _stage: None,
        nearest_projection_fn=lambda _p, _anchor: None,
        primary_hit_counter_key='edge_hit_count',
        collision_diagnostics=diag,
        max_hit_rows=[],
        wall_rows=wall_rows,
        coating_summary_rows=[],
        wall_law_counts={},
        wall_summary_counts={},
        stuck=np.zeros(1, dtype=bool),
        absorbed=np.zeros(1, dtype=bool),
        active=np.ones(1, dtype=bool),
        max_wall_hits_per_step=5,
        epsilon_offset_m=1.0e-6,
        on_boundary_tol_m=1.0e-6,
        triangle_surface_3d=None,
    )

    assert bool(result.invalid_mask_stopped) is True
    assert str(result.invalid_stop_reason) == 'collision_valid_mask_hard_invalid_retry_exhausted'
    assert result.position == pytest.approx([-0.999, 0.0])
    assert result.velocity == pytest.approx([4.0, 0.0])
    assert int(diag['invalid_mask_retry_count']) == 4
    assert int(diag['invalid_mask_retry_exhausted_count']) == 1
    assert int(diag['collision_reintegrated_segments_count']) == 1
    assert len(wall_rows) == 1


def test_collision_replay_allows_mixed_stencil_without_invalid_stop(monkeypatch: pytest.MonkeyPatch):
    diag = _collision_diag_stub()
    wall_rows: list[dict[str, object]] = []

    step = ProcessStepRow(step_id=1, step_name='run', start_s=0.0, end_s=1.0, output_segment_name='run')
    particles = SimpleNamespace(
        particle_id=np.asarray([1], dtype=np.int64),
        stick_probability=np.asarray([0.0], dtype=np.float64),
    )

    monkeypatch.setattr(
        'particle_tracer_unified.solvers.high_fidelity_collision.locate_physical_hit_state',
        lambda **kwargs: (
            BoundaryHit(
                position=np.asarray([-1.0, 0.0], dtype=np.float64),
                normal=np.asarray([-1.0, 0.0], dtype=np.float64),
                part_id=10,
                alpha_hint=0.1,
            ),
            np.asarray([-4.0, 0.0], dtype=np.float64),
            0.05,
        ),
    )
    monkeypatch.setattr(
        'particle_tracer_unified.solvers.high_fidelity_collision._apply_wall_hit_step',
        lambda **kwargs: (
            wall_rows.append({'particle_id': 1, 'part_id': int(kwargs['part_id'])}),
            WallHitStepResult(
                np.asarray([-0.999, 0.0], dtype=np.float64),
                np.asarray([4.0, 0.0], dtype=np.float64),
                0.55,
                1,
                1,
                False,
            ),
        )[1],
    )
    monkeypatch.setattr(
        'particle_tracer_unified.solvers.high_fidelity_collision.advance_freeflight_segment',
        lambda **kwargs: (
            np.asarray([-0.7, 0.0], dtype=np.float64),
            np.asarray([4.0, 0.0], dtype=np.float64),
            1,
            np.asarray([[-0.7, 0.0]], dtype=np.float64),
            VALID_MASK_STATUS_MIXED_STENCIL,
        ),
    )

    result = _advance_colliding_particle(
        runtime=SimpleNamespace(),
        step=step,
        particles=particles,
        particle_index=0,
        rng=np.random.default_rng(123),
        t=0.6,
        x_start=np.asarray([-0.8, 0.0], dtype=np.float64),
        v_start=np.asarray([-4.0, 0.0], dtype=np.float64),
        dt_step=0.6,
        spatial_dim=2,
        compiled={},
        integrator_mode=0,
        base_adaptive_substep_enabled=0,
        adaptive_substep_tau_ratio=0.5,
        adaptive_substep_max_splits=4,
        min_remaining_dt_ratio=0.0,
        tau_p_i=1.0,
        particle_diameter_i=1.0e-6,
        flow_scale_particle_i=1.0,
        drag_scale_particle_i=1.0,
        body_scale_particle_i=1.0,
        global_flow_scale=1.0,
        global_drag_tau_scale=1.0,
        global_body_accel_scale=1.0,
        body_accel=np.zeros(2, dtype=np.float64),
        min_tau_p_s=1.0e-9,
        gas_density_kgm3=1.2,
        gas_mu_pas=1.8e-5,
        drag_model_mode=0,
        valid_mask_retry_then_stop_enabled=True,
        initial_x_next=np.asarray([-3.2, 0.0], dtype=np.float64),
        initial_v_next=np.asarray([-4.0, 0.0], dtype=np.float64),
        initial_stage_points=np.asarray([[-3.2, 0.0]], dtype=np.float64),
        initial_valid_mask_status=VALID_MASK_STATUS_CLEAN,
        initial_primary_hit=BoundaryHit(
            position=np.asarray([-1.0, 0.0], dtype=np.float64),
            normal=np.asarray([-1.0, 0.0], dtype=np.float64),
            part_id=10,
            alpha_hint=0.1,
        ),
        initial_primary_hit_counted=False,
        inside_fn=lambda _p: True,
        strict_inside_fn=lambda _p: True,
        primary_hit_fn=lambda _p0, _stage: None,
        nearest_projection_fn=lambda _p, _anchor: None,
        primary_hit_counter_key='edge_hit_count',
        collision_diagnostics=diag,
        max_hit_rows=[],
        wall_rows=wall_rows,
        coating_summary_rows=[],
        wall_law_counts={},
        wall_summary_counts={},
        stuck=np.zeros(1, dtype=bool),
        absorbed=np.zeros(1, dtype=bool),
        active=np.ones(1, dtype=bool),
        max_wall_hits_per_step=5,
        epsilon_offset_m=1.0e-6,
        on_boundary_tol_m=1.0e-6,
        triangle_surface_3d=None,
    )

    assert bool(result.invalid_mask_stopped) is False
    assert int(result.valid_mask_status) == int(VALID_MASK_STATUS_MIXED_STENCIL)
    assert int(diag['invalid_mask_retry_count']) == 0
    assert int(diag['invalid_mask_retry_exhausted_count']) == 0
    assert len(wall_rows) == 1


def test_visualization_state_helpers_include_invalid_mask_stopped():
    final_df = pd.DataFrame(
        [
            {'particle_id': 1, 'active': 1, 'stuck': 0, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 0},
            {'particle_id': 2, 'active': 0, 'stuck': 0, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 1},
            {'particle_id': 3, 'active': 0, 'stuck': 1, 'absorbed': 0, 'escaped': 0, 'invalid_mask_stopped': 0},
        ]
    )
    labels = state_labels(final_df)
    counts = final_state_counts(final_df)
    step_df = pd.DataFrame(
        {
            'time_s': [0.1, 0.2, 0.3],
            'active_count': [3, 2, 1],
            'stuck_count': [0, 0, 1],
            'absorbed_count': [0, 0, 0],
            'escaped_count': [0, 0, 0],
            'invalid_mask_stopped_count_step': [0, 1, 0],
        }
    )

    assert labels.tolist() == ['active_free_flight', 'invalid_mask_stopped', 'stuck']
    assert counts['invalid_mask_stopped'] == 1
    assert step_state_count_series(step_df, 'invalid_mask_stopped').tolist() == pytest.approx([0.0, 1.0, 1.0])


def test_visualization_state_helpers_split_contact_from_free_flight():
    final_df = pd.DataFrame(
        [
            {'particle_id': 1, 'active': 1, 'contact_sliding': 0, 'contact_endpoint_stopped': 0},
            {'particle_id': 2, 'active': 1, 'contact_sliding': 1, 'contact_endpoint_stopped': 0},
            {'particle_id': 3, 'active': 1, 'contact_sliding': 1, 'contact_endpoint_stopped': 1},
        ]
    )
    labels = state_labels(final_df)
    counts = final_state_counts(final_df)

    assert labels.tolist() == ['active_free_flight', 'contact_sliding', 'contact_endpoint_stopped']
    assert counts['active_free_flight'] == 1
    assert counts['contact_sliding'] == 1
    assert counts['contact_endpoint_stopped'] == 1


def test_export_result_graphs_summary_includes_invalid_mask_stopped(tmp_path: Path):
    axes = np.linspace(-1.0, 1.0, 81)
    field_path = tmp_path / 'field_visual_invalid_stop.npz'
    _write_field_bundle(field_path, axes, axes)
    payload = {key: value for key, value in np.load(field_path).items()}
    payload['ux'] = 4.0 * np.ones((axes.size, axes.size), dtype=np.float64)
    payload['uy'] = np.zeros((axes.size, axes.size), dtype=np.float64)
    valid_mask = np.ones((axes.size, axes.size), dtype=bool)
    valid_mask[axes >= -0.5, :] = False
    payload['valid_mask'] = valid_mask
    np.savez_compressed(field_path, **payload)

    particles_path = tmp_path / 'particles_visual_invalid_stop.csv'
    pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': -0.8,
                'y': -0.2,
                'vx': 0.2,
                'vy': 0.0,
                'release_time': 0.0,
                'mass': 1e-15,
                'diameter': 1e-6,
                'density': 1200,
                'charge': 0,
                'source_part_id': 10,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    ).to_csv(particles_path, index=False)

    cfg_dir = tmp_path / 'cfg_visual_invalid_stop'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = _write_config(
        cfg_dir,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: (
            cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())}),
            cfg.setdefault('providers', {}).update(
                {
                    'field': {
                        'kind': 'precomputed_npz',
                        'npz_path': str(field_path.resolve()),
                    }
                }
            ),
            cfg.setdefault('solver', {}).update(
                {
                    't_end': 0.2,
                    'dt': 0.2,
                    'save_every': 1,
                    'integrator': 'etd2',
                    'adaptive_substep_max_splits': 4,
                    'valid_mask_policy': 'retry_then_stop',
                }
            ),
            cfg.setdefault('provider_contract', {}).update({'boundary_field_support': 'off'}),
        ),
    )

    out_dir = tmp_path / 'out_visual_invalid_stop'
    run_solver_2d_from_yaml(cfg, output_dir=out_dir)
    export_result_graphs(out_dir, case_dir=ROOT / 'examples' / 'minimal_2d', sample_trajectories=1)

    summary = json.loads((out_dir / 'visualizations' / 'graphs' / 'graph_summary.json').read_text(encoding='utf-8'))
    assert int(summary['final_state_counts']['invalid_mask_stopped']) == 1
    assert (out_dir / 'visualizations' / 'graphs' / '02_final_state_bar_and_pie.png').exists()


def test_export_boundary_diagnostics_reports_mixed_and_hard_invalid_regions(tmp_path: Path):
    case_dir = tmp_path / 'case'
    generated_dir = case_dir / 'generated'
    generated_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / 'run_output'
    output_dir.mkdir(parents=True, exist_ok=True)

    axis = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing='ij')
    valid_mask = np.asarray(
        [
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
        ],
        dtype=bool,
    )
    boundary_edges = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    mesh_vertices = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    mesh_quads = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
    np.savez_compressed(
        generated_dir / 'comsol_geometry_2d.npz',
        axis_0=axis,
        axis_1=axis,
        sdf=np.asarray(
            [
                [-0.2, -0.1, 0.1],
                [-0.1, -0.05, 0.2],
                [0.05, 0.2, 0.4],
            ],
            dtype=np.float64,
        ),
        normal_0=np.zeros_like(xx, dtype=np.float64),
        normal_1=np.ones_like(yy, dtype=np.float64),
        valid_mask=valid_mask,
        boundary_edges=boundary_edges,
        boundary_edge_part_ids=np.asarray([1, 2, 3, 4], dtype=np.int32),
        mesh_vertices=mesh_vertices,
        mesh_quads=mesh_quads,
    )
    np.savez_compressed(
        generated_dir / 'comsol_field_2d.npz',
        axis_0=axis,
        axis_1=axis,
        times=np.asarray([0.0], dtype=np.float64),
        valid_mask=valid_mask,
        ux=np.ones_like(xx, dtype=np.float64),
        uy=np.zeros_like(yy, dtype=np.float64),
    )
    pd.DataFrame(
        [
            {'particle_id': 1, 'x': 0.9, 'y': 0.9, 'invalid_mask_stopped': 1},
            {'particle_id': 2, 'x': 0.1, 'y': 0.1, 'invalid_mask_stopped': 0},
        ]
    ).to_csv(output_dir / 'final_particles.csv', index=False)

    boundary_dir = export_boundary_diagnostics(case_dir=case_dir, output_dir=output_dir)
    report = json.loads((boundary_dir / 'boundary_diagnostics_report.json').read_text(encoding='utf-8'))

    assert int(report['mixed_stencil_grid_count']) > 0
    assert int(report['hard_invalid_grid_count']) > 0
    assert int(report['invalid_mask_stopped_point_count']) == 1
    assert (boundary_dir / '06_mixed_stencil_hotspots.png').exists()
    assert (boundary_dir / '07_hard_invalid_stop_hotspots.png').exists()


def test_polyline_alpha_is_normalized_by_segment_index():
    assert normalize_polyline_alpha(0, 0.5, 2) == pytest.approx(0.25)
    assert normalize_polyline_alpha(1, 0.5, 2) == pytest.approx(0.75)
    assert normalize_polyline_alpha(1, -0.5, 2) == pytest.approx(0.5)
    assert normalize_polyline_alpha(1, 1.5, 2) == pytest.approx(1.0)


def test_polyline_edge_hit_uses_earliest_segment_and_normalized_alpha():
    from types import SimpleNamespace

    edges = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    geom = SimpleNamespace(
        spatial_dim=2,
        boundary_edges=edges,
        boundary_edge_part_ids=np.asarray([1, 2, 3, 4], dtype=np.int32),
    )
    runtime = SimpleNamespace(geometry_provider=SimpleNamespace(geometry=geom))

    p0 = np.asarray([0.5, 0.5], dtype=np.float64)
    stage_points = np.asarray([[1.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    hit = polyline_hit_from_boundary_edges(runtime, p0, stage_points)
    assert hit is not None
    assert isinstance(hit, BoundaryHit)
    assert hit.alpha_hint == pytest.approx(0.25)
    assert int(hit.part_id) == 2

    stage_points_second = np.asarray([[0.5, 0.5], [1.5, 0.5]], dtype=np.float64)
    hit_second = polyline_hit_from_boundary_edges(runtime, p0, stage_points_second)
    assert hit_second is not None
    assert isinstance(hit_second, BoundaryHit)
    assert hit_second.alpha_hint == pytest.approx(0.75)
    assert int(hit_second.part_id) == 2
    assert int(hit_second.primitive_id) == 1
    assert str(hit_second.primitive_kind) == 'edge'
    assert bool(hit_second.is_ambiguous) is False
    assert hit_second.local_signed_distance(np.asarray([0.5, 0.5], dtype=np.float64)) < 0.0


def test_batch_boundary_edge_hits_match_scalar_polyline_contract():
    from types import SimpleNamespace

    edges = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    geom = SimpleNamespace(
        spatial_dim=2,
        boundary_edges=edges,
        boundary_edge_part_ids=np.asarray([1, 2, 3, 4], dtype=np.int32),
    )
    runtime = SimpleNamespace(geometry_provider=SimpleNamespace(geometry=geom))
    starts = np.asarray([[0.5, 0.5], [0.5, 0.5], [0.2, 0.2]], dtype=np.float64)
    stages = np.asarray(
        [
            [[1.5, 0.5], [0.5, 0.5]],
            [[0.5, 0.5], [1.5, 0.5]],
            [[0.3, 0.2], [0.4, 0.2]],
        ],
        dtype=np.float64,
    )

    hits = polyline_hits_from_boundary_edges_batch(
        runtime,
        starts,
        stages,
        particle_indices=np.asarray([10, 11, 12], dtype=np.int64),
    )

    assert set(hits) == {10, 11}
    assert int(hits[10].part_id) == 2
    assert int(hits[10].primitive_id) == 1
    assert str(hits[10].primitive_kind) == 'edge'
    assert bool(hits[10].is_ambiguous) is False
    assert hits[10].alpha_hint == pytest.approx(0.25)
    assert int(hits[11].part_id) == 2
    assert int(hits[11].primitive_id) == 1
    assert hits[11].alpha_hint == pytest.approx(0.75)


def test_nearest_boundary_edge_features_report_part_and_distance():
    from types import SimpleNamespace

    edges = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    geom = SimpleNamespace(
        spatial_dim=2,
        boundary_edges=edges,
        boundary_edge_part_ids=np.asarray([1, 2, 3, 4], dtype=np.int32),
    )
    runtime = SimpleNamespace(geometry_provider=SimpleNamespace(geometry=geom))

    part_ids, distances = nearest_boundary_edge_features_2d(
        runtime,
        np.asarray([[0.95, 0.5], [0.5, 0.1]], dtype=np.float64),
    )

    assert part_ids.tolist() == [2, 1]
    assert distances[0] == pytest.approx(0.05)
    assert distances[1] == pytest.approx(0.1)


def test_boundary_service_2d_matches_loop_truth_and_edge_hit_contract():
    edges = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    loops = build_boundary_loops_2d(edges)
    runtime = SimpleNamespace(
        geometry_provider=SimpleNamespace(
            geometry=SimpleNamespace(
                spatial_dim=2,
                axes=(np.linspace(0.0, 1.0, 11), np.linspace(0.0, 1.0, 11)),
                boundary_edges=edges,
                boundary_edge_part_ids=np.asarray([10, 20, 30, 40], dtype=np.int32),
                boundary_loops_2d=loops,
                sdf=np.zeros((11, 11), dtype=np.float64),
                nearest_boundary_part_id_map=np.zeros((11, 11), dtype=np.int32),
                normal_components=(np.zeros((11, 11), dtype=np.float64), np.ones((11, 11), dtype=np.float64)),
            )
        ),
        field_provider=None,
    )
    service = build_boundary_service(runtime, spatial_dim=2, on_boundary_tol_m=1.0e-9, triangle_surface_3d=None)
    assert service.primary_hit_counter_key == 'edge_hit_count'
    assert bool(service.inside(np.asarray([0.5, 0.5], dtype=np.float64))) is True
    assert bool(service.inside(np.asarray([1.2, 0.5], dtype=np.float64))) is False
    hit = service.polyline_hit(
        np.asarray([0.5, 0.5], dtype=np.float64),
        np.asarray([[0.8, 0.5], [1.2, 0.5]], dtype=np.float64),
    )
    assert hit is not None
    assert int(hit.primitive_id) == 1
    assert str(hit.primitive_kind) == 'edge'
    assert bool(hit.is_ambiguous) is False
    endpoint_hit = service.segment_hit(
        np.asarray([0.5, 0.5], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64),
    )
    assert endpoint_hit is not None
    assert str(endpoint_hit.primitive_kind) == 'edge'
    assert bool(endpoint_hit.is_ambiguous) is True
    assert hit.position == pytest.approx([1.0, 0.5])
    assert hit.normal == pytest.approx([1.0, 0.0])
    assert hit.part_id == 20
    assert hit.alpha_hint == pytest.approx(0.75)


def test_etd_reduces_position_error_for_linear_drag_2d(tmp_path: Path):
    particle = pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': 0.0,
                'y': 0.0,
                'vx': 1.0,
                'vy': 0.0,
                'release_time': 0.0,
                'mass': 1e-15,
                'diameter': 1.0e-4,
                'density': 1000.0,
                'charge': 0.0,
                'source_part_id': 10,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    )
    particles_path = tmp_path / 'one_particle_2d.csv'
    particle.to_csv(particles_path, index=False)

    common_mutation = lambda cfg: (
        cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())}),
        cfg.setdefault('providers', {}).setdefault('geometry', {}).update({'bounds': [-10.0, 10.0, -10.0, 10.0], 'grid_shape': [51, 51]}),
        cfg.setdefault('providers', {}).setdefault('field', {}).update({'shear_rate': 0.0}),
        cfg.setdefault('solver', {}).update({'dt': 0.05, 't_end': 0.2, 'save_every': 1, 'min_tau_p_s': 1.0e-8}),
    )

    drag_cfg_dir = tmp_path / 'drag_cfg'
    etd_cfg_dir = tmp_path / 'etd_cfg'
    drag_cfg_dir.mkdir(parents=True, exist_ok=True)
    etd_cfg_dir.mkdir(parents=True, exist_ok=True)

    drag_cfg = _write_config(
        drag_cfg_dir,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: (common_mutation(cfg), cfg.setdefault('solver', {}).update({'integrator': 'drag_relaxation'})),
    )
    etd_cfg = _write_config(
        etd_cfg_dir,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: (common_mutation(cfg), cfg.setdefault('solver', {}).update({'integrator': 'etd'})),
    )
    drag_out = tmp_path / 'out_drag_2d'
    etd_out = tmp_path / 'out_etd_2d'
    run_solver_2d_from_yaml(drag_cfg, output_dir=drag_out)
    run_solver_2d_from_yaml(etd_cfg, output_dir=etd_out)

    mu = 1.8e-5
    tau = 1000.0 * (1.0e-4 ** 2) / (18.0 * mu)
    t_end = 0.2
    exact_x = tau * (1.0 - np.exp(-t_end / tau))

    x_drag = float(pd.read_csv(drag_out / 'final_particles.csv').loc[0, 'x'])
    x_etd = float(pd.read_csv(etd_out / 'final_particles.csv').loc[0, 'x'])
    err_drag = abs(x_drag - exact_x)
    err_etd = abs(x_etd - exact_x)
    assert err_etd < err_drag


def test_etd2_is_not_worse_than_etd_vs_fine_reference_2d(tmp_path: Path):
    particle = pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': 0.0,
                'y': 0.2,
                'vx': 0.8,
                'vy': 0.0,
                'release_time': 0.0,
                'mass': 1e-15,
                'diameter': 1.0e-5,
                'density': 1000.0,
                'charge': 0.0,
                'source_part_id': 10,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    )
    particles_path = tmp_path / 'one_particle_etd2_2d.csv'
    particle.to_csv(particles_path, index=False)

    def _mutate(cfg, integrator: str, dt: float):
        cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())})
        cfg.setdefault('providers', {}).setdefault('geometry', {}).update({'bounds': [-10.0, 10.0, -10.0, 10.0], 'grid_shape': [81, 81]})
        cfg.setdefault('providers', {}).setdefault('field', {}).update({'shear_rate': 4.0})
        cfg.setdefault('solver', {}).update({'integrator': integrator, 'dt': dt, 't_end': 0.2, 'save_every': 1, 'min_tau_p_s': 1.0e-8})

    cfg_dir = tmp_path / 'etd2_compare_2d'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_etd_dir = cfg_dir / 'etd'
    cfg_etd2_dir = cfg_dir / 'etd2'
    cfg_ref_dir = cfg_dir / 'ref'
    cfg_etd_dir.mkdir(parents=True, exist_ok=True)
    cfg_etd2_dir.mkdir(parents=True, exist_ok=True)
    cfg_ref_dir.mkdir(parents=True, exist_ok=True)
    cfg_etd = _write_config(cfg_etd_dir, ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml', mutate=lambda cfg: _mutate(cfg, 'etd', 0.05))
    cfg_etd2 = _write_config(cfg_etd2_dir, ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml', mutate=lambda cfg: _mutate(cfg, 'etd2', 0.05))
    cfg_ref = _write_config(cfg_ref_dir, ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml', mutate=lambda cfg: _mutate(cfg, 'etd2', 0.0025))

    out_etd = tmp_path / 'out_etd_cmp_2d'
    out_etd2 = tmp_path / 'out_etd2_cmp_2d'
    out_ref = tmp_path / 'out_ref_cmp_2d'
    run_solver_2d_from_yaml(cfg_etd, output_dir=out_etd)
    run_solver_2d_from_yaml(cfg_etd2, output_dir=out_etd2)
    run_solver_2d_from_yaml(cfg_ref, output_dir=out_ref)

    p_etd = pd.read_csv(out_etd / 'final_particles.csv').loc[0, ['x', 'y']].to_numpy(dtype=float)
    p_etd2 = pd.read_csv(out_etd2 / 'final_particles.csv').loc[0, ['x', 'y']].to_numpy(dtype=float)
    p_ref = pd.read_csv(out_ref / 'final_particles.csv').loc[0, ['x', 'y']].to_numpy(dtype=float)
    err_etd = float(np.linalg.norm(p_etd - p_ref))
    err_etd2 = float(np.linalg.norm(p_etd2 - p_ref))
    assert err_etd2 <= err_etd + 1e-12


def test_etd_reduces_position_error_for_linear_drag_3d(tmp_path: Path):
    particle = pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'vx': 1.0,
                'vy': 0.0,
                'vz': 0.0,
                'release_time': 0.0,
                'mass': 1e-15,
                'diameter': 1.0e-4,
                'density': 1000.0,
                'charge': 0.0,
                'source_part_id': 10,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    )
    particles_path = tmp_path / 'one_particle_3d.csv'
    particle.to_csv(particles_path, index=False)

    common_mutation = lambda cfg: (
        cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())}),
        cfg.setdefault('providers', {}).setdefault('geometry', {}).update(
            {'bounds': [-10.0, 10.0, -10.0, 10.0, -10.0, 10.0], 'grid_shape': [31, 31, 31]}
        ),
        cfg.setdefault('providers', {}).setdefault('field', {}).update({'shear_rate': 0.0}),
        cfg.setdefault('solver', {}).update({'dt': 0.05, 't_end': 0.2, 'save_every': 1, 'min_tau_p_s': 1.0e-8}),
    )

    drag_cfg_dir = tmp_path / 'drag_cfg_3d'
    etd_cfg_dir = tmp_path / 'etd_cfg_3d'
    drag_cfg_dir.mkdir(parents=True, exist_ok=True)
    etd_cfg_dir.mkdir(parents=True, exist_ok=True)

    drag_cfg = _write_config(
        drag_cfg_dir,
        ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml',
        mutate=lambda cfg: (common_mutation(cfg), cfg.setdefault('solver', {}).update({'integrator': 'drag_relaxation'})),
    )
    etd_cfg = _write_config(
        etd_cfg_dir,
        ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml',
        mutate=lambda cfg: (common_mutation(cfg), cfg.setdefault('solver', {}).update({'integrator': 'etd'})),
    )
    drag_out = tmp_path / 'out_drag_3d'
    etd_out = tmp_path / 'out_etd_3d'
    run_solver_3d_from_yaml(drag_cfg, output_dir=drag_out)
    run_solver_3d_from_yaml(etd_cfg, output_dir=etd_out)

    mu = 1.8e-5
    tau = 1000.0 * (1.0e-4 ** 2) / (18.0 * mu)
    t_end = 0.2
    exact_x = tau * (1.0 - np.exp(-t_end / tau))

    x_drag = float(pd.read_csv(drag_out / 'final_particles.csv').loc[0, 'x'])
    x_etd = float(pd.read_csv(etd_out / 'final_particles.csv').loc[0, 'x'])
    err_drag = abs(x_drag - exact_x)
    err_etd = abs(x_etd - exact_x)
    assert err_etd < err_drag


def test_etd2_is_not_worse_than_etd_vs_fine_reference_3d(tmp_path: Path):
    particle = pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': 0.0,
                'y': 0.2,
                'z': -0.1,
                'vx': 0.8,
                'vy': 0.0,
                'vz': 0.0,
                'release_time': 0.0,
                'mass': 1e-15,
                'diameter': 1.0e-5,
                'density': 1000.0,
                'charge': 0.0,
                'source_part_id': 10,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    )
    particles_path = tmp_path / 'one_particle_etd2_3d.csv'
    particle.to_csv(particles_path, index=False)

    def _mutate(cfg, integrator: str, dt: float):
        cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())})
        cfg.setdefault('providers', {}).setdefault('geometry', {}).update({'bounds': [-10.0, 10.0, -10.0, 10.0, -10.0, 10.0], 'grid_shape': [41, 41, 41]})
        cfg.setdefault('providers', {}).setdefault('field', {}).update({'shear_rate': 5.0})
        cfg.setdefault('solver', {}).update({'integrator': integrator, 'dt': dt, 't_end': 0.2, 'save_every': 1, 'min_tau_p_s': 1.0e-8})

    cfg_dir = tmp_path / 'etd2_compare_3d'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_etd_dir = cfg_dir / 'etd'
    cfg_etd2_dir = cfg_dir / 'etd2'
    cfg_ref_dir = cfg_dir / 'ref'
    cfg_etd_dir.mkdir(parents=True, exist_ok=True)
    cfg_etd2_dir.mkdir(parents=True, exist_ok=True)
    cfg_ref_dir.mkdir(parents=True, exist_ok=True)
    cfg_etd = _write_config(cfg_etd_dir, ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml', mutate=lambda cfg: _mutate(cfg, 'etd', 0.05))
    cfg_etd2 = _write_config(cfg_etd2_dir, ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml', mutate=lambda cfg: _mutate(cfg, 'etd2', 0.05))
    cfg_ref = _write_config(cfg_ref_dir, ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml', mutate=lambda cfg: _mutate(cfg, 'etd2', 0.0025))

    out_etd = tmp_path / 'out_etd_cmp_3d'
    out_etd2 = tmp_path / 'out_etd2_cmp_3d'
    out_ref = tmp_path / 'out_ref_cmp_3d'
    run_solver_3d_from_yaml(cfg_etd, output_dir=out_etd)
    run_solver_3d_from_yaml(cfg_etd2, output_dir=out_etd2)
    run_solver_3d_from_yaml(cfg_ref, output_dir=out_ref)

    p_etd = pd.read_csv(out_etd / 'final_particles.csv').loc[0, ['x', 'y', 'z']].to_numpy(dtype=float)
    p_etd2 = pd.read_csv(out_etd2 / 'final_particles.csv').loc[0, ['x', 'y', 'z']].to_numpy(dtype=float)
    p_ref = pd.read_csv(out_ref / 'final_particles.csv').loc[0, ['x', 'y', 'z']].to_numpy(dtype=float)
    err_etd = float(np.linalg.norm(p_etd - p_ref))
    err_etd2 = float(np.linalg.norm(p_etd2 - p_ref))
    assert err_etd2 <= err_etd + 1e-12


def _single_reflection_expected_zero_flow(*, x0: float, wall_x: float, v0: float, tau: float, dt: float) -> tuple[float, float]:
    travel = float(wall_x - x0)
    decay_hit = 1.0 - travel / (float(v0) * float(tau))
    t_hit = -float(tau) * float(np.log(decay_hit))
    v_hit = float(v0) * decay_hit
    remaining = float(dt) - t_hit
    decay_rem = float(np.exp(-remaining / float(tau)))
    x_final = float(wall_x) - v_hit * float(tau) * (1.0 - decay_rem)
    v_final = -v_hit * decay_rem
    return x_final, v_final


def test_2d_single_wall_reflection_uses_physical_hit_velocity_and_time(tmp_path: Path):
    particle = pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': 0.9,
                'y': 0.0,
                'vx': 2.0,
                'vy': 0.0,
                'release_time': 0.0,
                'mass': 1e-15,
                'diameter': 1.8e-4,
                'density': 1000.0,
                'charge': 0.0,
                'source_part_id': 10,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    )
    particles_path = tmp_path / 'single_reflection_2d.csv'
    particle.to_csv(particles_path, index=False)
    part_walls = pd.DataFrame(
        [
            {'part_id': 10, 'part_name': 'left_bottom', 'material_id': 1, 'material_name': 'steel', 'wall_law': 'specular', 'wall_restitution': 1.0, 'wall_diffuse_fraction': 0.0, 'wall_stick_probability': 0.0},
            {'part_id': 20, 'part_name': 'right_top', 'material_id': 2, 'material_name': 'ceramic', 'wall_law': 'specular', 'wall_restitution': 1.0, 'wall_diffuse_fraction': 0.0, 'wall_stick_probability': 0.0},
        ]
    )
    part_walls_path = tmp_path / 'part_walls_reflect.csv'
    part_walls.to_csv(part_walls_path, index=False)

    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: (
            cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve()), 'part_walls_csv': str(part_walls_path.resolve())}),
            cfg.setdefault('providers', {}).setdefault('field', {}).update({'shear_rate': 0.0}),
            cfg.setdefault('solver', {}).update({'integrator': 'etd', 'dt': 0.2, 't_end': 0.2, 'save_every': 1, 'min_tau_p_s': 1.0e-8}),
        ),
    )
    out_dir = tmp_path / 'out_single_reflection_2d'
    run_solver_2d_from_yaml(config_path, output_dir=out_dir)

    final_df = pd.read_csv(out_dir / 'final_particles.csv')
    x_final = float(final_df.loc[0, 'x'])
    vx_final = float(final_df.loc[0, 'v_x'])
    expected_x, expected_vx = _single_reflection_expected_zero_flow(x0=0.9, wall_x=1.0, v0=2.0, tau=0.1, dt=0.2)
    assert x_final == pytest.approx(expected_x, abs=2.0e-4)
    assert vx_final == pytest.approx(expected_vx, abs=2.0e-4)


def test_3d_single_wall_reflection_uses_physical_hit_velocity_and_time(tmp_path: Path):
    particle = pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': 0.9,
                'y': 0.0,
                'z': 0.0,
                'vx': 2.0,
                'vy': 0.0,
                'vz': 0.0,
                'release_time': 0.0,
                'mass': 1e-15,
                'diameter': 1.8e-4,
                'density': 1000.0,
                'charge': 0.0,
                'source_part_id': 10,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    )
    particles_path = tmp_path / 'single_reflection_3d.csv'
    particle.to_csv(particles_path, index=False)
    part_walls = pd.DataFrame(
        [
            {'part_id': 10, 'part_name': 'wall_10', 'material_id': 1, 'material_name': 'steel', 'wall_law': 'specular', 'wall_restitution': 1.0, 'wall_diffuse_fraction': 0.0, 'wall_stick_probability': 0.0},
            {'part_id': 20, 'part_name': 'wall_20', 'material_id': 2, 'material_name': 'ceramic', 'wall_law': 'specular', 'wall_restitution': 1.0, 'wall_diffuse_fraction': 0.0, 'wall_stick_probability': 0.0},
        ]
    )
    part_walls_path = tmp_path / 'part_walls_reflect_3d.csv'
    part_walls.to_csv(part_walls_path, index=False)

    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml',
        mutate=lambda cfg: (
            cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve()), 'part_walls_csv': str(part_walls_path.resolve())}),
            cfg.setdefault('providers', {}).setdefault('field', {}).update({'shear_rate': 0.0}),
            cfg.setdefault('solver', {}).update({'integrator': 'etd', 'dt': 0.2, 't_end': 0.2, 'save_every': 1, 'min_tau_p_s': 1.0e-8}),
        ),
    )
    out_dir = tmp_path / 'out_single_reflection_3d'
    run_solver_3d_from_yaml(config_path, output_dir=out_dir)

    final_df = pd.read_csv(out_dir / 'final_particles.csv')
    x_final = float(final_df.loc[0, 'x'])
    vx_final = float(final_df.loc[0, 'v_x'])
    expected_x, expected_vx = _single_reflection_expected_zero_flow(x0=0.9, wall_x=1.0, v0=2.0, tau=0.1, dt=0.2)
    assert x_final == pytest.approx(expected_x, abs=2.0e-4)
    assert vx_final == pytest.approx(expected_vx, abs=2.0e-4)


def test_apply_wall_hit_step_subtracts_hit_time_across_multiple_hits():
    runtime = SimpleNamespace(
        wall_catalog=None,
        geometry_provider=None,
        field_provider=SimpleNamespace(field=SimpleNamespace(axes=(np.asarray([0.0, 1.0], dtype=np.float64), np.asarray([0.0, 1.0], dtype=np.float64)))),
    )
    step = ProcessStepRow(step_id=1, step_name='run', start_s=0.0, end_s=1.0)
    particles = SimpleNamespace(
        particle_id=np.asarray([1], dtype=np.int64),
        stick_probability=np.asarray([0.0], dtype=np.float64),
    )
    collision_diagnostics = _initial_collision_diagnostics()
    max_hit_rows: list[dict[str, object]] = []
    wall_rows: list[dict[str, object]] = []
    wall_law_counts: dict[str, int] = {}
    wall_summary_counts: dict[tuple[int, str, str], int] = {}
    stuck = np.asarray([False], dtype=bool)
    absorbed = np.asarray([False], dtype=bool)
    active = np.asarray([True], dtype=bool)

    result1 = _apply_wall_hit_step(
        runtime=runtime,
        step=step,
        particles=particles,
        particle_index=0,
        rng=np.random.default_rng(123),
        hit=np.asarray([0.5, 0.5], dtype=np.float64),
        n_out=np.asarray([1.0, 0.0], dtype=np.float64),
        hit_dt=0.02,
        part_id=0,
        v_hit=np.asarray([1.0, 0.0], dtype=np.float64),
        remaining_dt=0.2,
        segment_dt=0.2,
        hit_count=0,
        total_hit_count=0,
        hit_part_ids=[],
        hit_outcomes=[],
        collision_diagnostics=collision_diagnostics,
        max_hit_rows=max_hit_rows,
        wall_rows=wall_rows,
        coating_summary_rows=[],
        wall_law_counts=wall_law_counts,
        wall_summary_counts=wall_summary_counts,
        stuck=stuck,
        absorbed=absorbed,
        active=active,
        max_wall_hits_per_step=5,
        min_remaining_dt=0.0,
        epsilon_offset_m=1.0e-6,
        on_boundary_tol_m=1.0e-6,
        t=0.2,
        triangle_surface_3d=None,
    )
    x1 = result1.position
    v1 = result1.velocity
    assert result1.should_break is False
    assert result1.entered_contact is False
    assert result1.remaining_dt == pytest.approx(0.18, abs=1e-15)

    result2 = _apply_wall_hit_step(
        runtime=runtime,
        step=step,
        particles=particles,
        particle_index=0,
        rng=np.random.default_rng(123),
        hit=np.asarray([0.4, 0.5], dtype=np.float64),
        n_out=np.asarray([-1.0, 0.0], dtype=np.float64),
        hit_dt=0.02,
        part_id=0,
        v_hit=v1,
        remaining_dt=result1.remaining_dt,
        segment_dt=result1.remaining_dt,
        hit_count=result1.hit_count,
        total_hit_count=result1.total_hit_count,
        hit_part_ids=[0],
        hit_outcomes=['reflected_specular'],
        collision_diagnostics=collision_diagnostics,
        max_hit_rows=max_hit_rows,
        wall_rows=wall_rows,
        coating_summary_rows=[],
        wall_law_counts=wall_law_counts,
        wall_summary_counts=wall_summary_counts,
        stuck=stuck,
        absorbed=absorbed,
        active=active,
        max_wall_hits_per_step=5,
        min_remaining_dt=0.0,
        epsilon_offset_m=1.0e-6,
        on_boundary_tol_m=1.0e-6,
        t=0.2,
        triangle_surface_3d=None,
    )
    assert result2.should_break is False
    assert result2.entered_contact is False
    assert result2.remaining_dt == pytest.approx(0.16, abs=1e-15)


def test_apply_wall_hit_step_records_minimal_max_hit_diagnostics():
    runtime = SimpleNamespace(
        wall_catalog=None,
        geometry_provider=None,
        field_provider=SimpleNamespace(
            field=SimpleNamespace(
                axes=(np.asarray([0.0, 1.0], dtype=np.float64), np.asarray([0.0, 1.0], dtype=np.float64))
            )
        ),
    )
    step = ProcessStepRow(step_id=1, step_name='run', start_s=0.0, end_s=1.0)
    particles = SimpleNamespace(
        particle_id=np.asarray([42], dtype=np.int64),
        stick_probability=np.asarray([0.0], dtype=np.float64),
    )
    collision_diagnostics = _initial_collision_diagnostics()
    max_hit_rows: list[dict[str, object]] = []
    hit_part_ids: list[int] = []
    hit_outcomes: list[str] = []
    stuck = np.asarray([False], dtype=bool)
    absorbed = np.asarray([False], dtype=bool)
    active = np.asarray([True], dtype=bool)

    result = _apply_wall_hit_step(
        runtime=runtime,
        step=step,
        particles=particles,
        particle_index=0,
        rng=np.random.default_rng(123),
        hit=np.asarray([0.0, 0.5], dtype=np.float64),
        n_out=np.asarray([-1.0, 0.0], dtype=np.float64),
        hit_dt=0.0,
        part_id=7,
        v_hit=np.asarray([-2.0, 0.0], dtype=np.float64),
        remaining_dt=0.2,
        segment_dt=0.2,
        hit_count=0,
        total_hit_count=0,
        hit_part_ids=hit_part_ids,
        hit_outcomes=hit_outcomes,
        collision_diagnostics=collision_diagnostics,
        max_hit_rows=max_hit_rows,
        wall_rows=[],
        coating_summary_rows=[],
        wall_law_counts={},
        wall_summary_counts={},
        stuck=stuck,
        absorbed=absorbed,
        active=active,
        max_wall_hits_per_step=1,
        min_remaining_dt=0.0,
        epsilon_offset_m=1.0e-6,
        on_boundary_tol_m=1.0e-6,
        t=0.2,
        triangle_surface_3d=None,
    )

    assert result.should_break is True
    assert result.entered_contact is False
    assert result.remaining_dt > 0.0
    assert result.hit_count == 1
    assert result.total_hit_count == 1
    assert int(collision_diagnostics['max_hits_reached_count']) == 1
    assert int(collision_diagnostics['max_hit_same_wall_count']) == 1
    assert int(collision_diagnostics['max_hit_multi_wall_count']) == 0
    assert collision_diagnostics['max_hit_last_part_counts'] == {'part=7': 1}
    assert collision_diagnostics['max_hit_last_outcome_counts'] == {'reflected_specular': 1}
    assert max_hit_rows[0]['particle_id'] == 42
    assert max_hit_rows[0]['part_id_sequence'] == '7'


def test_apply_wall_hit_step_converts_repeated_same_wall_hit_to_contact_sliding():
    runtime = SimpleNamespace(
        wall_catalog=None,
        geometry_provider=None,
        field_provider=SimpleNamespace(
            field=SimpleNamespace(
                axes=(np.asarray([0.0, 1.0], dtype=np.float64), np.asarray([0.0, 1.0], dtype=np.float64))
            )
        ),
    )
    step = ProcessStepRow(step_id=1, step_name='run', start_s=0.0, end_s=1.0)
    particles = SimpleNamespace(
        particle_id=np.asarray([42], dtype=np.int64),
        stick_probability=np.asarray([0.0], dtype=np.float64),
    )
    collision_diagnostics = _initial_collision_diagnostics()
    hit_part_ids = [7]
    hit_outcomes = ['reflected_specular']
    stuck = np.asarray([False], dtype=bool)
    absorbed = np.asarray([False], dtype=bool)
    active = np.asarray([True], dtype=bool)

    result = _apply_wall_hit_step(
        runtime=runtime,
        step=step,
        particles=particles,
        particle_index=0,
        rng=np.random.default_rng(123),
        hit=np.asarray([0.0, 0.5], dtype=np.float64),
        n_out=np.asarray([-1.0, 0.0], dtype=np.float64),
        hit_dt=0.0,
        part_id=7,
        v_hit=np.asarray([-2.0, 1.0], dtype=np.float64),
        remaining_dt=0.2,
        segment_dt=0.2,
        hit_count=1,
        total_hit_count=1,
        hit_part_ids=hit_part_ids,
        hit_outcomes=hit_outcomes,
        collision_diagnostics=collision_diagnostics,
        max_hit_rows=[],
        wall_rows=[],
        coating_summary_rows=[],
        wall_law_counts={},
        wall_summary_counts={},
        stuck=stuck,
        absorbed=absorbed,
        active=active,
        max_wall_hits_per_step=2,
        min_remaining_dt=0.0,
        epsilon_offset_m=1.0e-6,
        on_boundary_tol_m=1.0e-6,
        t=0.2,
        triangle_surface_3d=None,
    )

    assert result.should_break is True
    assert result.entered_contact is True
    assert result.remaining_dt == pytest.approx(0.0, abs=1e-15)
    assert result.hit_count == 2
    assert result.total_hit_count == 2
    assert int(collision_diagnostics['max_hits_reached_count']) == 0
    assert int(collision_diagnostics['contact_sliding_count']) == 1
    assert int(collision_diagnostics['contact_sliding_same_wall_count']) == 1
    assert collision_diagnostics['contact_sliding_part_counts'] == {'part=7': 1}
    assert result.contact_part_id == 7
    assert result.contact_normal == pytest.approx([-1.0, 0.0], abs=1e-15)
    assert np.dot(result.velocity, np.asarray([-1.0, 0.0], dtype=np.float64)) == pytest.approx(0.0, abs=1e-15)


def test_apply_wall_hit_step_converts_repeated_same_wall_hit_to_contact_sliding_3d():
    runtime = SimpleNamespace(
        wall_catalog=None,
        geometry_provider=None,
        field_provider=SimpleNamespace(
            field=SimpleNamespace(
                axes=(
                    np.asarray([0.0, 1.0], dtype=np.float64),
                    np.asarray([0.0, 1.0], dtype=np.float64),
                    np.asarray([0.0, 1.0], dtype=np.float64),
                )
            )
        ),
    )
    step = ProcessStepRow(step_id=1, step_name='run', start_s=0.0, end_s=1.0)
    particles = SimpleNamespace(
        particle_id=np.asarray([42], dtype=np.int64),
        stick_probability=np.asarray([0.0], dtype=np.float64),
    )
    collision_diagnostics = _initial_collision_diagnostics()
    hit_part_ids = [7]
    hit_outcomes = ['reflected_specular']
    stuck = np.asarray([False], dtype=bool)
    absorbed = np.asarray([False], dtype=bool)
    active = np.asarray([True], dtype=bool)

    result = _apply_wall_hit_step(
        runtime=runtime,
        step=step,
        particles=particles,
        particle_index=0,
        rng=np.random.default_rng(123),
        hit=np.asarray([0.0, 0.5, 0.5], dtype=np.float64),
        n_out=np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
        hit_dt=0.0,
        part_id=7,
        v_hit=np.asarray([-2.0, 1.0, 0.5], dtype=np.float64),
        remaining_dt=0.2,
        segment_dt=0.2,
        hit_count=1,
        total_hit_count=1,
        hit_part_ids=hit_part_ids,
        hit_outcomes=hit_outcomes,
        collision_diagnostics=collision_diagnostics,
        max_hit_rows=[],
        wall_rows=[],
        coating_summary_rows=[],
        wall_law_counts={},
        wall_summary_counts={},
        stuck=stuck,
        absorbed=absorbed,
        active=active,
        max_wall_hits_per_step=2,
        min_remaining_dt=0.0,
        epsilon_offset_m=1.0e-6,
        on_boundary_tol_m=1.0e-6,
        t=0.2,
        triangle_surface_3d=None,
    )

    assert result.should_break is True
    assert result.entered_contact is True
    assert result.remaining_dt == pytest.approx(0.0, abs=1e-15)
    assert int(collision_diagnostics['max_hits_reached_count']) == 0
    assert int(collision_diagnostics['contact_sliding_count']) == 1
    assert result.contact_part_id == 7
    assert result.contact_normal == pytest.approx([-1.0, 0.0, 0.0], abs=1e-15)
    assert np.dot(result.velocity, np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)) == pytest.approx(0.0, abs=1e-15)


def test_transient_endpoint_contact_releases_when_force_points_inside():
    axes = (
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )
    valid_mask = np.ones((3, 3), dtype=bool)
    times = np.asarray([0.0, 1.0], dtype=np.float64)
    zero = np.zeros((2, 3, 3), dtype=np.float64)
    ex = np.stack((np.zeros((3, 3), dtype=np.float64), np.full((3, 3), 10.0, dtype=np.float64)), axis=0)
    quantities = {
        'ux': zero,
        'uy': zero,
        'E_x': ex,
        'E_y': zero,
    }
    field_provider = FieldProviderND(
        field=RegularFieldND(
            spatial_dim=2,
            coordinate_system='cartesian_xy',
            axis_names=('x', 'y'),
            axes=axes,
            quantities={
                name: QuantitySeriesND(name=name, unit='', times=times, data=value, metadata={})
                for name, value in quantities.items()
            },
            valid_mask=valid_mask,
            time_mode='transient',
            metadata={'provider_kind': 'precomputed_npz'},
        ),
        kind='precomputed_npz',
    )
    boundary_edges = np.asarray(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    geometry = GeometryND(
        spatial_dim=2,
        coordinate_system='cartesian_xy',
        axes=axes,
        valid_mask=valid_mask,
        sdf=-np.ones((3, 3), dtype=np.float64),
        normal_components=(np.zeros((3, 3), dtype=np.float64), np.ones((3, 3), dtype=np.float64)),
        nearest_boundary_part_id_map=np.ones((3, 3), dtype=np.int32),
        boundary_edges=boundary_edges,
        boundary_edge_part_ids=np.asarray([1, 2, 3, 4], dtype=np.int32),
        boundary_loops_2d=build_boundary_loops_2d(boundary_edges),
    )
    runtime = SimpleNamespace(
        geometry_provider=GeometryProviderND(geometry=geometry, kind='test'),
        field_provider=field_provider,
    )
    compiled = _compile_runtime_arrays(runtime, spatial_dim=2)
    options = SolverRuntimeOptions(
        dt=0.01,
        t_end=0.01,
        base_save_every=1,
        plot_limit=0,
        rng_seed=1,
        max_wall_hits_per_step=2,
        min_remaining_dt_ratio=0.0,
        adaptive_substep_enabled=0,
        adaptive_substep_tau_ratio=1.0,
        adaptive_substep_max_splits=0,
        epsilon_offset_m=1.0e-6,
        on_boundary_tol_m=1.0e-6,
        write_collision_diagnostics=1,
        valid_mask_policy='retry_then_stop',
        output_options=RuntimeOutputOptions(),
        drag_model_mode=DRAG_MODEL_STOKES,
        drag_model_name='stokes',
        contact_tangent_motion_enabled=True,
    )
    state = RuntimeState(
        x=np.asarray([[1.0e-6, 0.0]], dtype=np.float64),
        v=np.zeros((1, 2), dtype=np.float64),
        released=np.asarray([True], dtype=bool),
        active=np.asarray([True], dtype=bool),
        stuck=np.asarray([False], dtype=bool),
        absorbed=np.asarray([False], dtype=bool),
        contact_sliding=np.asarray([True], dtype=bool),
        contact_endpoint_stopped=np.asarray([True], dtype=bool),
        contact_edge_index=np.asarray([0], dtype=np.int32),
        contact_part_id=np.asarray([1], dtype=np.int32),
        contact_normal=np.asarray([[-1.0, 0.0]], dtype=np.float64),
        escaped=np.asarray([False], dtype=bool),
        invalid_mask_stopped=np.asarray([False], dtype=bool),
        numerical_boundary_stopped=np.asarray([False], dtype=bool),
        invalid_stop_reason_code=np.zeros(1, dtype=np.uint8),
        save_positions=[],
        save_meta=[],
        wall_rows=[],
        coating_summary_rows=[],
        max_hit_rows=[],
        step_rows=[],
        wall_law_counts={},
        wall_summary_counts={},
        collision_diagnostics=_initial_collision_diagnostics(),
        rng=np.random.default_rng(1),
        prev_step_name=None,
        step_local_counter=0,
        save_index=0,
        x_trial=np.zeros((1, 2), dtype=np.float64),
        v_trial=np.zeros((1, 2), dtype=np.float64),
        x_mid_trial=np.zeros((1, 2), dtype=np.float64),
        substep_counts=np.ones(1, dtype=np.int32),
        valid_mask_status_flags=np.zeros(1, dtype=np.uint8),
        valid_mask_mixed_seen=np.asarray([False], dtype=bool),
        valid_mask_hard_seen=np.asarray([False], dtype=bool),
    )

    _advance_contact_sliding_particles_2d(
        runtime=runtime,
        state=state,
        options=options,
        compiled=compiled,
        boundary_service=build_boundary_service(runtime, spatial_dim=2, on_boundary_tol_m=1.0e-6, triangle_surface_3d=None),
        tau_p=np.asarray([1.0e-3], dtype=np.float64),
        particle_diameter=np.asarray([1.0e-6], dtype=np.float64),
        flow_scale_particle=np.asarray([1.0], dtype=np.float64),
        drag_scale_particle=np.asarray([1.0], dtype=np.float64),
        body_scale_particle=np.asarray([1.0], dtype=np.float64),
        phys={
            'flow_scale': 1.0,
            'drag_tau_scale': 1.0,
            'body_accel_scale': 1.0,
            'min_tau_p_s': 1.0e-6,
            'gas_density_kgm3': 1.2,
            'gas_mu_pas': 1.8e-5,
        },
        body_accel=np.zeros(2, dtype=np.float64),
        dt_step=0.01,
        t_next=1.0,
        electric_q_over_m_particle=np.asarray([1.0], dtype=np.float64),
    )

    assert bool(state.contact_sliding[0]) is False
    assert bool(state.contact_endpoint_stopped[0]) is False
    assert int(state.contact_edge_index[0]) == -1
    assert int(state.collision_diagnostics['contact_release_count']) == 1
    assert int(state.collision_diagnostics['contact_release_probe_reject_count']) == 0


def test_3d_contact_sliding_advances_on_triangle_face():
    axes = tuple(np.asarray([0.0, 0.5, 1.0], dtype=np.float64) for _ in range(3))
    valid_mask = np.ones((3, 3, 3), dtype=bool)
    quantities = {
        'ux': np.zeros_like(valid_mask, dtype=np.float64),
        'uy': np.zeros_like(valid_mask, dtype=np.float64),
        'uz': np.zeros_like(valid_mask, dtype=np.float64),
    }
    field_provider = _regular_field_provider_from_arrays(axes, valid_mask, quantities)
    triangles = _cube_triangles_oriented()
    geometry = GeometryND(
        spatial_dim=3,
        coordinate_system='cartesian_xyz',
        axes=axes,
        valid_mask=valid_mask,
        sdf=-np.ones((3, 3, 3), dtype=np.float64),
        normal_components=tuple(np.zeros((3, 3, 3), dtype=np.float64) for _ in range(3)),
        nearest_boundary_part_id_map=np.ones((3, 3, 3), dtype=np.int32),
        boundary_triangles=triangles,
        boundary_triangle_part_ids=np.ones(triangles.shape[0], dtype=np.int32),
    )
    runtime = SimpleNamespace(
        geometry_provider=GeometryProviderND(geometry=geometry, kind='test'),
        field_provider=field_provider,
    )
    compiled = _compile_runtime_arrays(runtime, spatial_dim=3)
    surface = build_triangle_surface(triangles, np.ones(triangles.shape[0], dtype=np.int32))
    boundary_service = build_boundary_service(
        runtime,
        spatial_dim=3,
        on_boundary_tol_m=1.0e-6,
        triangle_surface_3d=surface,
    )
    options = SolverRuntimeOptions(
        dt=0.1,
        t_end=0.1,
        base_save_every=1,
        plot_limit=0,
        rng_seed=1,
        max_wall_hits_per_step=2,
        min_remaining_dt_ratio=0.0,
        adaptive_substep_enabled=0,
        adaptive_substep_tau_ratio=1.0,
        adaptive_substep_max_splits=0,
        epsilon_offset_m=1.0e-6,
        on_boundary_tol_m=1.0e-6,
        write_collision_diagnostics=1,
        valid_mask_policy='retry_then_stop',
        output_options=RuntimeOutputOptions(),
        drag_model_mode=DRAG_MODEL_STOKES,
        drag_model_name='stokes',
        contact_tangent_motion_enabled=True,
    )
    state = RuntimeState(
        x=np.asarray([[1.0e-6, 0.5, 0.5]], dtype=np.float64),
        v=np.asarray([[0.0, 0.1, 0.0]], dtype=np.float64),
        released=np.asarray([True], dtype=bool),
        active=np.asarray([True], dtype=bool),
        stuck=np.asarray([False], dtype=bool),
        absorbed=np.asarray([False], dtype=bool),
        contact_sliding=np.asarray([True], dtype=bool),
        contact_endpoint_stopped=np.asarray([False], dtype=bool),
        contact_edge_index=np.asarray([-1], dtype=np.int32),
        contact_part_id=np.asarray([1], dtype=np.int32),
        contact_normal=np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float64),
        escaped=np.asarray([False], dtype=bool),
        invalid_mask_stopped=np.asarray([False], dtype=bool),
        numerical_boundary_stopped=np.asarray([False], dtype=bool),
        invalid_stop_reason_code=np.zeros(1, dtype=np.uint8),
        save_positions=[],
        save_meta=[],
        wall_rows=[],
        coating_summary_rows=[],
        max_hit_rows=[],
        step_rows=[],
        wall_law_counts={},
        wall_summary_counts={},
        collision_diagnostics=_initial_collision_diagnostics(),
        rng=np.random.default_rng(1),
        prev_step_name=None,
        step_local_counter=0,
        save_index=0,
        x_trial=np.zeros((1, 3), dtype=np.float64),
        v_trial=np.zeros((1, 3), dtype=np.float64),
        x_mid_trial=np.zeros((1, 3), dtype=np.float64),
        substep_counts=np.ones(1, dtype=np.int32),
        valid_mask_status_flags=np.zeros(1, dtype=np.uint8),
        valid_mask_mixed_seen=np.asarray([False], dtype=bool),
        valid_mask_hard_seen=np.asarray([False], dtype=bool),
    )

    _advance_contact_sliding_particles_3d(
        runtime=runtime,
        state=state,
        options=options,
        compiled=compiled,
        boundary_service=boundary_service,
        tau_p=np.asarray([1.0], dtype=np.float64),
        particle_diameter=np.asarray([1.0e-6], dtype=np.float64),
        flow_scale_particle=np.asarray([1.0], dtype=np.float64),
        drag_scale_particle=np.asarray([1.0], dtype=np.float64),
        body_scale_particle=np.asarray([1.0], dtype=np.float64),
        phys={
            'flow_scale': 1.0,
            'drag_tau_scale': 1.0,
            'body_accel_scale': 1.0,
            'min_tau_p_s': 1.0e-6,
            'gas_density_kgm3': 1.2,
            'gas_mu_pas': 1.8e-5,
        },
        body_accel=np.zeros(3, dtype=np.float64),
        dt_step=0.1,
        t_next=0.1,
    )

    assert bool(state.contact_sliding[0]) is True
    assert int(state.contact_edge_index[0]) >= 0
    assert state.x[0, 0] == pytest.approx(1.0e-6, abs=1.0e-12)
    assert state.x[0, 1] > 0.5
    assert int(state.collision_diagnostics['contact_tangent_step_count']) == 1
    assert int(state.collision_diagnostics['contact_valid_mask_reject_count']) == 0


def test_final_snapshot_matches_t_end_when_not_divisible(tmp_path: Path):
    out_dir = tmp_path / 'out_2d'
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('solver', {}).update({'dt': 0.3, 't_end': 1.0, 'save_every': 1, 'plot_particle_limit': 3}),
    )
    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    save_frames = pd.read_csv(out_dir / 'save_frames.csv')
    assert save_frames['time_s'].iloc[-1] == pytest.approx(1.0, abs=1e-12)


def test_plot_particle_limit_zero_skips_trajectory_plot(tmp_path: Path):
    out_dir = tmp_path / 'out_2d_no_plot'
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('solver', {}).update({'plot_particle_limit': 0}),
    )
    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    assert not (out_dir / 'trajectories.png').exists()


def test_3d_wall_events_use_boundary_part_ids(tmp_path: Path):
    out_dir = tmp_path / 'out_3d'
    config_path = _write_config(tmp_path, ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml')
    run_solver_3d_from_yaml(config_path, output_dir=out_dir)
    wall_events = pd.read_csv(out_dir / 'wall_events.csv')
    assert not wall_events.empty
    part_ids = set(int(v) for v in wall_events['part_id'].tolist())
    assert part_ids.issubset({10, 20})
    assert 1 not in part_ids
    assert {
        'hit_time_s',
        'hit_x_m',
        'normal_x',
        'v_hit_x_mps',
        'impact_angle_deg_from_normal',
        'boundary_primitive_id',
        'boundary_primitive_kind',
        'boundary_hit_ambiguous',
    }.issubset(wall_events.columns)
    assert np.all(wall_events['boundary_primitive_id'].to_numpy(dtype=np.int64) >= 0)
    assert np.all(wall_events['hit_time_s'].to_numpy(dtype=np.float64) <= wall_events['time_s'].to_numpy(dtype=np.float64) + 1.0e-12)
    assert np.all(np.isfinite(wall_events['impact_speed_mps'].to_numpy(dtype=np.float64)))
    coating_summary = pd.read_csv(out_dir / 'coating_summary_by_part.csv')
    assert {'impact_count', 'stuck_count', 'deposited_mass_kg', 'mean_impact_angle_deg_from_normal'}.issubset(coating_summary.columns)
    assert int(coating_summary['impact_count'].sum()) == int(len(wall_events))


def test_validate_closed_surface_triangles_rejects_open_surface():
    tri = _cube_triangles_oriented()[:-1]
    with pytest.raises(ValueError, match='closed 2-manifold'):
        validate_closed_surface_triangles(tri)


def test_validate_closed_surface_triangles_rejects_orientation_mismatch():
    tri = _cube_triangles_oriented().copy()
    tri[0] = tri[0][[0, 2, 1], :]
    with pytest.raises(ValueError, match='orientation mismatch'):
        validate_closed_surface_triangles(tri)


def test_3d_boundary_points_are_inside_for_closed_surface():
    tri = _cube_triangles_oriented()
    surface = build_triangle_surface(tri, np.ones(tri.shape[0], dtype=np.int32), validate_closed=True)
    inside_mid, on_mid = point_inside_surface(surface, np.asarray([0.0, 0.0, 0.0], dtype=np.float64), on_boundary_tol=1.0e-8)
    inside_edge, on_edge = point_inside_surface(surface, np.asarray([1.0, 0.3, -0.2], dtype=np.float64), on_boundary_tol=1.0e-7)
    outside, on_out = point_inside_surface(surface, np.asarray([1.2, 0.0, 0.0], dtype=np.float64), on_boundary_tol=1.0e-7)
    assert bool(inside_mid) and not bool(on_mid)
    assert bool(inside_edge) and bool(on_edge)
    assert not bool(outside) and not bool(on_out)


def test_boundary_service_3d_matches_closed_surface_truth():
    tri = _cube_triangles_oriented()
    surface = build_triangle_surface(tri, np.ones(tri.shape[0], dtype=np.int32), validate_closed=True)
    runtime = SimpleNamespace(
        geometry_provider=SimpleNamespace(
            geometry=SimpleNamespace(
                spatial_dim=3,
                axes=(
                    np.linspace(-1.0, 1.0, 9),
                    np.linspace(-1.0, 1.0, 9),
                    np.linspace(-1.0, 1.0, 9),
                ),
                boundary_loops_2d=(),
                boundary_triangles=tri,
                boundary_triangle_part_ids=np.ones(tri.shape[0], dtype=np.int32),
                sdf=np.zeros((9, 9, 9), dtype=np.float64),
                nearest_boundary_part_id_map=np.ones((9, 9, 9), dtype=np.int32),
                normal_components=(
                    np.zeros((9, 9, 9), dtype=np.float64),
                    np.zeros((9, 9, 9), dtype=np.float64),
                    np.ones((9, 9, 9), dtype=np.float64),
                ),
            )
        ),
        field_provider=None,
    )
    service = build_boundary_service(runtime, spatial_dim=3, on_boundary_tol_m=1.0e-7, triangle_surface_3d=surface)
    assert service.primary_hit_counter_key == 'triangle_hit_count'
    assert bool(service.inside(np.asarray([0.0, 0.0, 0.0], dtype=np.float64))) is True
    assert bool(service.inside(np.asarray([1.3, 0.0, 0.0], dtype=np.float64))) is False
    hit = service.polyline_hit(
        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64),
    )
    assert hit is not None
    assert hit.position == pytest.approx([1.0, 0.0, 0.0], abs=1.0e-8)
    assert hit.normal == pytest.approx([1.0, 0.0, 0.0], abs=1.0e-8)
    assert hit.part_id == 1
    assert hit.alpha_hint == pytest.approx(0.75, abs=1.0e-6)
    assert int(hit.primitive_id) >= 0
    assert str(hit.primitive_kind) == 'triangle'
    assert abs(hit.local_signed_distance(np.asarray([0.0, 0.0, 0.0], dtype=np.float64))) > 0.0


def test_trial_collision_batch_prefetches_boundary_hits_in_3d():
    tri = _cube_triangles_oriented()
    surface = build_triangle_surface(tri, np.ones(tri.shape[0], dtype=np.int32), validate_closed=True)
    runtime = SimpleNamespace(
        geometry_provider=SimpleNamespace(
            geometry=SimpleNamespace(
                spatial_dim=3,
                axes=(
                    np.linspace(-1.0, 1.0, 9),
                    np.linspace(-1.0, 1.0, 9),
                    np.linspace(-1.0, 1.0, 9),
                ),
                boundary_loops_2d=(),
                boundary_triangles=tri,
                boundary_triangle_part_ids=np.ones(tri.shape[0], dtype=np.int32),
                sdf=np.zeros((9, 9, 9), dtype=np.float64),
                nearest_boundary_part_id_map=np.ones((9, 9, 9), dtype=np.int32),
                normal_components=(
                    np.zeros((9, 9, 9), dtype=np.float64),
                    np.zeros((9, 9, 9), dtype=np.float64),
                    np.ones((9, 9, 9), dtype=np.float64),
                ),
            )
        ),
        field_provider=None,
    )
    service = build_boundary_service(runtime, spatial_dim=3, on_boundary_tol_m=1.0e-7, triangle_surface_3d=surface)
    batch = _classify_trial_collisions(
        runtime,
        spatial_dim=3,
        n_particles=1,
        active=np.asarray([True], dtype=bool),
        x=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        x_trial=np.asarray([[1.5, 0.0, 0.0]], dtype=np.float64),
        x_mid_trial=np.asarray([[0.5, 0.0, 0.0]], dtype=np.float64),
        integrator_mode=1,
        boundary_service=service,
        on_boundary_tol_m=1.0e-7,
        collision_diagnostics={
            'on_boundary_promoted_inside_count': 0,
            'etd2_midpoint_outside_count': 0,
        },
    )
    assert batch.colliders.tolist() == [0]
    assert batch.safe.size == 0
    assert 0 in batch.prefetched_hits
    assert isinstance(batch.prefetched_hits[0], BoundaryHit)
    assert batch.prefetched_hits[0].position == pytest.approx([1.0, 0.0, 0.0], abs=1.0e-8)


def test_trial_collision_batch_prefetches_2d_inside_to_inside_wall_crossing():
    outer = np.asarray(
        [
            [[0.0, 0.0], [4.0, 0.0]],
            [[4.0, 0.0], [4.0, 4.0]],
            [[4.0, 4.0], [0.0, 4.0]],
            [[0.0, 4.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    inner = np.asarray(
        [
            [[1.5, 1.5], [2.5, 1.5]],
            [[2.5, 1.5], [2.5, 2.5]],
            [[2.5, 2.5], [1.5, 2.5]],
            [[1.5, 2.5], [1.5, 1.5]],
        ],
        dtype=np.float64,
    )
    edges = np.concatenate((outer, inner), axis=0)
    runtime = SimpleNamespace(
        geometry_provider=SimpleNamespace(
            geometry=SimpleNamespace(
                spatial_dim=2,
                axes=(np.asarray([0.0, 4.0], dtype=np.float64), np.asarray([0.0, 4.0], dtype=np.float64)),
                boundary_loops_2d=build_boundary_loops_2d(edges),
                boundary_edges=edges,
                boundary_edge_part_ids=np.asarray([10] * 4 + [20] * 4, dtype=np.int32),
                sdf=np.zeros((2, 2), dtype=np.float64),
                nearest_boundary_part_id_map=np.zeros((2, 2), dtype=np.int32),
                normal_components=(np.zeros((2, 2), dtype=np.float64), np.ones((2, 2), dtype=np.float64)),
            )
        ),
        field_provider=None,
    )
    service = build_boundary_service(runtime, spatial_dim=2, on_boundary_tol_m=1.0e-9, triangle_surface_3d=None)
    batch = _classify_trial_collisions(
        runtime,
        spatial_dim=2,
        n_particles=1,
        active=np.asarray([True], dtype=bool),
        x=np.asarray([[0.5, 2.0]], dtype=np.float64),
        x_trial=np.asarray([[3.5, 2.0]], dtype=np.float64),
        x_mid_trial=np.asarray([[3.5, 2.0]], dtype=np.float64),
        integrator_mode=int(get_integrator_spec('drag_relaxation').mode),
        boundary_service=service,
        on_boundary_tol_m=1.0e-9,
        collision_diagnostics={
            'on_boundary_promoted_inside_count': 0,
            'etd2_midpoint_outside_count': 0,
        },
    )

    assert batch.colliders.tolist() == [0]
    assert batch.safe.size == 0
    assert 0 in batch.prefetched_hits
    assert int(batch.prefetched_hits[0].part_id) == 20
    assert batch.prefetched_hits[0].position == pytest.approx([1.5, 2.0], abs=1.0e-8)


def test_dimension_wrappers_match_shared_solver_entrypoint():
    prepared_2d = build_prepared_runtime_2d(ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml')
    shared_2d = build_prepared_runtime_for_dim(ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml', spatial_dim=2)
    prepared_3d = build_prepared_runtime_3d(ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml')
    shared_3d = build_prepared_runtime_for_dim(ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml', spatial_dim=3)

    assert int(prepared_2d.runtime.spatial_dim) == 2
    assert int(shared_2d.runtime.spatial_dim) == 2
    assert int(prepared_3d.runtime.spatial_dim) == 3
    assert int(shared_3d.runtime.spatial_dim) == 3


def test_run_prepared_runtime_can_skip_file_outputs():
    prepared = build_prepared_runtime_from_yaml(ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml')

    report = run_prepared_runtime(prepared, output_dir=None, spatial_dim=2)

    assert int(report['outputs_written']) == 0
    assert int(report['save_frame_count']) == 0
    assert report['positions_file'] == ''
    assert report['wall_summary_file'] == ''
    assert report['runtime_step_summary_file'] == ''
    assert int(report['particle_count']) == int(prepared.runtime.particles.count)


def test_output_artifact_mode_minimal_skips_bulk_outputs(tmp_path: Path):
    out_dir = tmp_path / 'out_2d_minimal'
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.update({'output': {'artifact_mode': 'minimal'}}),
    )

    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    report = json.loads((out_dir / 'solver_report.json').read_text(encoding='utf-8'))

    assert (out_dir / 'final_particles.csv').exists()
    assert (out_dir / 'wall_summary.json').exists()
    assert (out_dir / 'coating_summary_by_part.csv').exists()
    coating_summary = pd.read_csv(out_dir / 'coating_summary_by_part.csv')
    if not coating_summary.empty:
        assert coating_summary['mean_impact_speed_mps'].notna().any()
    assert not (out_dir / 'positions_2d.npy').exists()
    assert not (out_dir / 'save_frames.csv').exists()
    assert not (out_dir / 'wall_events.csv').exists()
    assert not (out_dir / 'runtime_step_summary.csv').exists()
    assert not (out_dir / 'prepared_runtime_summary.json').exists()
    assert not (out_dir / 'resolved_particles.csv').exists()
    assert not (out_dir / 'trajectories.png').exists()
    assert str(report['positions_file']) == ''
    assert str(report['runtime_step_summary_file']) == ''
    assert str(report['coating_summary_file']) == 'coating_summary_by_part.csv'


def test_solver_reports_schiller_naumann_drag_model(tmp_path: Path):
    out_dir = tmp_path / 'out_2d_schiller_naumann'
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('solver', {}).update(
            {'drag_model': 'schiller_naumann', 't_end': 0.05, 'save_every': 1, 'plot_particle_limit': 0}
        ),
    )

    report, _prepared = run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    solver_report = json.loads((out_dir / 'solver_report.json').read_text(encoding='utf-8'))
    diag = json.loads((out_dir / 'collision_diagnostics.json').read_text(encoding='utf-8'))

    assert report['drag_model'] == 'schiller_naumann'
    assert solver_report['drag_model'] == 'schiller_naumann'
    assert diag['drag_model'] == 'schiller_naumann'


def test_segment_output_filenames_are_sanitized(tmp_path: Path):
    steps = pd.DataFrame(
        [
            {
                'step_id': 1,
                'step_name': 'etch',
                'start_s': 0.0,
                'end_s': 1.0,
                'output_segment_name': 'etch:phase/1',
            }
        ]
    )
    steps_path = tmp_path / 'process_steps_sanitized.csv'
    steps.to_csv(steps_path, index=False)
    out_dir = tmp_path / 'out_2d_segment_names'
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('paths', {}).update({'process_steps_csv': str(steps_path.resolve())}),
    )

    run_solver_2d_from_yaml(config_path, output_dir=out_dir)

    assert (out_dir / 'segments' / 'positions_etch_phase_1_2d.npy').exists()


def test_single_run_segment_does_not_duplicate_positions_output(tmp_path: Path):
    out_dir = tmp_path / 'out_single_segment_no_duplicate'
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('solver', {}).update({'t_end': 0.02, 'save_every': 1, 'plot_particle_limit': 0}),
    )

    run_solver_2d_from_yaml(config_path, output_dir=out_dir)

    assert (out_dir / 'positions_2d.npy').exists()
    assert (out_dir / 'save_frames.csv').exists()
    assert (out_dir / 'segment_summary.csv').exists()
    assert not (out_dir / 'segments' / 'positions_run_2d.npy').exists()


def test_3d_collision_diagnostics_and_max_hits_limit_are_applied(tmp_path: Path):
    particles = pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'vx': 80.0,
                'vy': 0.0,
                'vz': 0.0,
                'release_time': 0.0,
                'mass': 1e-15,
                'diameter': 1e-6,
                'density': 1200.0,
                'charge': 0.0,
                'source_part_id': 10,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    )
    particles_path = tmp_path / 'fast_particles_3d.csv'
    particles.to_csv(particles_path, index=False)
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml',
        mutate=lambda cfg: (
            cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())}),
            cfg.setdefault('solver', {}).update(
                {
                    'dt': 0.2,
                    't_end': 0.2,
                    'save_every': 1,
                    'min_tau_p_s': 1.0,
                    'max_wall_hits_per_step': 1,
                    'min_remaining_dt_ratio': 0.0,
                }
            ),
            cfg.setdefault('output', {}).update({'write_collision_diagnostics': 1}),
        ),
    )
    out_dir = tmp_path / 'out_diag_hits_3d'
    run_solver_3d_from_yaml(config_path, output_dir=out_dir)
    diag = json.loads((out_dir / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    assert int(diag['max_wall_hits_per_step']) == 1
    assert int(diag['triangle_hit_count']) >= 1
    assert int(diag['max_hits_reached_count']) >= 1
    assert int(diag['max_hit_event_summary']['event_count']) == int(diag['max_hits_reached_count'])
    assert int(diag['max_hit_event_summary']['unique_particle_count']) >= 1
    max_hit_events = pd.read_csv(out_dir / 'max_hit_events.csv')
    assert not max_hit_events.empty
    assert set(max_hit_events.columns).issuperset({'time_s', 'particle_id', 'hits_in_step', 'remaining_dt_s', 'part_id_sequence'})


def test_3d_geometry_truth_keeps_non_escaped_particles_inside_surface(tmp_path: Path):
    out_dir = tmp_path / 'out_3d_inside_truth'
    report, prepared = run_solver_3d_from_yaml(ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml', output_dir=out_dir)
    geom = prepared.runtime.geometry_provider.geometry
    assert geom.boundary_triangles is not None
    surface = build_triangle_surface(
        np.asarray(geom.boundary_triangles, dtype=np.float64),
        np.asarray(geom.boundary_triangle_part_ids, dtype=np.int32),
        validate_closed=True,
    )
    final_df = pd.read_csv(out_dir / 'final_particles.csv')
    pts = final_df.loc[final_df['escaped'] == 0, ['x', 'y', 'z']].to_numpy(dtype=np.float64)
    inside = [point_inside_surface(surface, p, on_boundary_tol=2.0e-6)[0] for p in pts]
    assert bool(np.all(np.asarray(inside, dtype=bool)))


def test_comsol_precomputed_case_passes_strict_provider_contract(tmp_path: Path):
    cfg = ROOT / 'examples' / 'comsol_from_data_2d' / 'run_config.yaml'
    out_dir = tmp_path / 'out_comsol_provider_contract'
    prepared = build_prepared_runtime_from_yaml(cfg)
    report = build_boundary_field_support_report(prepared)
    assert report['passed'] is True
    assert report['status_counts']['non_clean'] == 0
    run_solver_2d_from_yaml(cfg, output_dir=out_dir)
    assert (out_dir / 'provider_contract_report.json').exists()
    assert (out_dir / 'solver_report.json').exists()


def test_production_config_list_excludes_triangle_mesh_stable_profile():
    assert not (ROOT / 'examples' / 'comsol_from_data_2d_10k' / 'run_config_prod_etd2_mesh_stable.yaml').exists()


@pytest.mark.parametrize(
    'config_name',
    [
        'run_config_prod_etd2_base.yaml',
    ],
)
def test_production_configs_use_minimal_artifacts_with_collision_diagnostics(config_name: str):
    cfg = yaml.safe_load((ROOT / 'examples' / 'comsol_from_data_2d_10k' / config_name).read_text(encoding='utf-8'))
    output_cfg = cfg.get('output', {})
    assert output_cfg.get('artifact_mode') == 'minimal'
    assert int(output_cfg.get('write_collision_diagnostics', 0)) == 1


def test_triangle_mesh_field_backend_runs_in_2d_solver(tmp_path: Path):
    mesh_field_path = _write_triangle_mesh_field_npz(tmp_path / 'field_mesh.npz')
    particles = pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': 0.25,
                'y': 0.25,
                'vx': 0.0,
                'vy': 0.0,
                'release_time': 0.0,
                'mass': 1e-15,
                'diameter': 1e-6,
                'density': 1200.0,
                'charge': 0.0,
                'source_part_id': 1,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    )
    particles_path = tmp_path / 'particles_mesh.csv'
    particles.to_csv(particles_path, index=False)
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: (
            cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())}),
            cfg.setdefault('providers', {}).setdefault('geometry', {}).update({'bounds': [0.0, 1.0, 0.0, 1.0], 'grid_shape': [41, 41]}),
            cfg.setdefault('providers', {}).update({'field': {'kind': 'precomputed_triangle_mesh_npz', 'npz_path': str(mesh_field_path.resolve())}}),
            cfg.setdefault('solver', {}).update({'dt': 0.05, 't_end': 0.05, 'save_every': 1, 'integrator': 'etd2'}),
            cfg.setdefault('output', {}).update({'write_collision_diagnostics': 1}),
        ),
    )
    out_dir = tmp_path / 'out_mesh_backend'
    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    report = json.loads((out_dir / 'solver_report.json').read_text(encoding='utf-8'))
    final_df = pd.read_csv(out_dir / 'final_particles.csv')
    assert str(report['field_backend_kind']) == 'triangle_mesh_2d'
    assert int(final_df['invalid_mask_stopped'].sum()) == 0


def test_comsol_builder_geometry_only_writes_geometry_without_run_config(tmp_path: Path):
    out_dir = tmp_path / 'comsol_case_geom_only'
    write_case_files(
        ROOT / 'data' / 'argon_gec_ccp_base2.mphtxt',
        out_dir,
        geometry_only=True,
        diagnostic_grid_spacing_m=1.0e-3,
    )
    assert (out_dir / 'generated' / 'comsol_geometry_2d.npz').exists()
    assert not (out_dir / 'generated' / 'comsol_field_2d.npz').exists()
    assert not (out_dir / 'run_config.yaml').exists()


def test_comsol_builder_does_not_write_triangle_mesh_field_by_default(tmp_path: Path):
    out_dir = tmp_path / 'comsol_case_mesh'
    write_case_files(
        ROOT / 'data' / 'argon_gec_ccp_base2.mphtxt',
        out_dir,
        field_bundle_path=ROOT / 'data' / 'regridded_repo_field_bundle_argon_gec_ccp_base2_2d.npz',
        diagnostic_grid_spacing_m=5.0e-4,
    )
    assert (out_dir / 'run_config.yaml').exists()
    cfg = yaml.safe_load((out_dir / 'run_config.yaml').read_text(encoding='utf-8'))
    assert 'source_events_csv' not in cfg.get('paths', {})
    assert cfg.get('input_contract', {}).get('initial_particle_field_support') == 'strict'
    assert not (out_dir / 'source_events.csv').exists()
    summary = json.loads((out_dir / 'generated' / 'comsol_case_summary.json').read_text(encoding='utf-8'))
    assert summary['provider_contract']['passed'] is True
    assert int(summary['provider_contract']['non_clean']) == 0
    assert (out_dir / 'generated' / 'provider_contract_report.json').exists()
    assert not (out_dir / 'generated' / 'provider_boundary_summary.csv').exists()
    assert 'provider_boundary_summary' not in summary['generated_files']
    assert summary['field_summary']['geometry_mask_applied'] is False
    assert int(summary['field_summary']['field_ghost_cells']) == 8
    assert summary['field_summary']['field_valid_mask_source'] == 'bundle_valid_mask_and_finite_field_quantities'
    assert int(summary['field_summary']['field_valid_node_count']) <= int(summary['field_summary']['finite_field_node_count'])
    assert int(summary['field_summary']['field_valid_node_count']) > int(summary['field_summary']['geometry_valid_node_count'])
    assert int(summary['field_summary']['provider_support_expanded_node_count']) > 0
    assert int(summary['field_summary']['provider_support_removed_nonfinite_node_count']) == 0
    assert int(summary['field_summary']['field_valid_node_count']) >= int(summary['field_summary']['particle_release_valid_node_count'])
    assert summary['field_summary']['support_phi_quality'] is None
    geometry_quality = summary['field_summary']['geometry_sdf_quality_against_field_valid_mask']
    assert int(geometry_quality['nonfinite_node_count']) == 0
    assert int(geometry_quality['grid_node_count']) == int(geometry_quality['finite_node_count'])
    assert 'inside_nonpositive_count' in geometry_quality
    assert 'outside_positive_count' in geometry_quality
    with np.load(out_dir / 'generated' / 'comsol_field_2d.npz') as payload:
        assert 'support_phi' not in payload.files
    assert not (out_dir / 'generated' / 'comsol_field_mesh_2d.npz').exists()
    assert not (out_dir / 'run_config_mesh.yaml').exists()


def test_comsol_builder_particles_only_generates_clean_release_domain(tmp_path: Path):
    out_dir = tmp_path / 'comsol_case_clean_particles'
    write_case_files(
        ROOT / 'data' / 'argon_gec_ccp_base2.mphtxt',
        out_dir,
        field_bundle_path=ROOT / 'data' / 'regridded_repo_field_bundle_argon_gec_ccp_base2_2d.npz',
        diagnostic_grid_spacing_m=5.0e-4,
    )
    write_particles_for_case(
        ROOT / 'data' / 'argon_gec_ccp_base2.mphtxt',
        out_dir,
        particle_count=128,
        release_span_s=0.4,
        seed=123,
        min_release_offset_cells=2.0,
    )
    particles = pd.read_csv(out_dir / 'particles.csv')
    assert len(particles) == 128
    assert particles['release_time'].iloc[0] == pytest.approx(0.0)
    assert particles['release_time'].iloc[-1] == pytest.approx(0.4)
    assert particles['release_offset_m'].min() > 0.0
    prepared = build_prepared_runtime_from_yaml(out_dir / 'run_config.yaml')
    report = build_initial_particle_field_support_report(prepared)
    assert report['status_counts']['non_clean'] == 0
    assert particles['source_part_id'].nunique() > 1


def test_comsol_builder_requires_field_bundle_for_runnable_case(tmp_path: Path):
    with pytest.raises(ValueError, match='requires --field-bundle'):
        write_case_files(ROOT / 'data' / 'argon_gec_ccp_base2.mphtxt', tmp_path / 'missing_bundle_case')


def test_comsol_builder_rejects_axis_mismatch_bundle(tmp_path: Path):
    mesh = parse_comsol_mphtxt(ROOT / 'data' / 'argon_gec_ccp_base2.mphtxt')
    arrays = build_precomputed_arrays(mesh, diagnostic_grid_spacing_m=1.0e-3)
    bundle_path = _write_field_bundle(tmp_path / 'bad_bundle.npz', arrays['axes_x'], arrays['axes_y'], axis_0_shift=1.0e-4)
    with pytest.raises(ValueError, match='axis_0'):
        write_case_files(
            ROOT / 'data' / 'argon_gec_ccp_base2.mphtxt',
            tmp_path / 'bad_bundle_case',
            field_bundle_path=bundle_path,
            diagnostic_grid_spacing_m=1.0e-3,
        )


def test_comsol_builder_rejects_bundle_missing_velocity_components(tmp_path: Path):
    mesh = parse_comsol_mphtxt(ROOT / 'data' / 'argon_gec_ccp_base2.mphtxt')
    arrays = build_precomputed_arrays(mesh, diagnostic_grid_spacing_m=1.0e-3)
    shape = (arrays['axes_x'].size, arrays['axes_y'].size)
    bundle_path = tmp_path / 'missing_ux_bundle.npz'
    np.savez_compressed(
        bundle_path,
        axis_0=arrays['axes_x'],
        axis_1=arrays['axes_y'],
        times=np.asarray([0.0], dtype=np.float64),
        valid_mask=np.ones(shape, dtype=bool),
        uy=np.zeros(shape, dtype=np.float64),
        mu=np.ones(shape, dtype=np.float64) * 1.8e-5,
    )
    with pytest.raises(ValueError, match='ux and uy'):
        write_case_files(
            ROOT / 'data' / 'argon_gec_ccp_base2.mphtxt',
            tmp_path / 'missing_ux_case',
            field_bundle_path=bundle_path,
            diagnostic_grid_spacing_m=1.0e-3,
        )


def test_merge_near_duplicate_axis_collapses_fp_noise():
    axis = np.asarray([0.0, 1e-16, 2e-16, 0.0254, 0.0254 + 5e-13, 0.1], dtype=np.float64)
    merged = _merge_near_duplicate_axis(axis, atol=1e-12)
    assert merged.shape == (3,)
    assert np.min(np.diff(merged)) > 1e-12
    assert merged[0] == pytest.approx(1e-16, abs=1e-15)
    assert merged[1] == pytest.approx(0.0254, abs=1e-12)


def test_order_quad_vertices_removes_bow_tie_ordering():
    vertices = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    raw = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    ordered = _order_quad_vertices(vertices, raw)
    poly = vertices[ordered[0]]
    area = 0.5 * abs(sum(poly[i, 0] * poly[(i + 1) % 4, 1] - poly[(i + 1) % 4, 0] * poly[i, 1] for i in range(4)))
    assert area == pytest.approx(1.0)


def test_comsol_boundary_edges_preserve_closed_mphtxt_boundary():
    mesh = parse_comsol_mphtxt(ROOT / 'data' / 'argon_gec_ccp_base2.mphtxt')
    arrays = build_precomputed_arrays(mesh)
    total_edg = int(mesh.type_blocks['edg'].elements.shape[0])
    preserved = int(arrays['boundary_edges'].shape[0])
    assert preserved == total_edg
    unique_parts = set(int(v) for v in np.unique(arrays['boundary_part_ids']))
    assert all(v > 0 for v in unique_parts)
    assert np.min(np.diff(arrays['axes_x'])) > 1e-12
    assert np.min(np.diff(arrays['axes_y'])) > 1e-12
    rounded_vertices = []
    for seg in np.asarray(arrays['boundary_edges'], dtype=np.float64):
        rounded_vertices.append(tuple(np.round(seg[0], 12)))
        rounded_vertices.append(tuple(np.round(seg[1], 12)))
    degree_counts = Counter(rounded_vertices)
    assert set(degree_counts.values()) == {2}


def test_boundary_loops_reconstruct_inside_outside_truth():
    edges = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    loops = build_boundary_loops_2d(edges)
    pts = np.asarray([[0.5, 0.5], [1.5, 0.5], [0.2, 0.8]], dtype=np.float64)
    inside = points_inside_boundary_loops_2d(pts, loops)
    assert [bool(v) for v in inside] == [True, False, True]


def test_boundary_loops_support_nested_hole_truth():
    edges = np.asarray(
        [
            [[0.0, 0.0], [4.0, 0.0]],
            [[4.0, 0.0], [4.0, 4.0]],
            [[4.0, 4.0], [0.0, 4.0]],
            [[0.0, 4.0], [0.0, 0.0]],
            [[1.0, 1.0], [3.0, 1.0]],
            [[3.0, 1.0], [3.0, 3.0]],
            [[3.0, 3.0], [1.0, 3.0]],
            [[1.0, 3.0], [1.0, 1.0]],
        ],
        dtype=np.float64,
    )
    loops = build_boundary_loops_2d(edges)
    assert len(loops) == 2
    pts = np.asarray(
        [
            [0.5, 0.5],
            [2.0, 2.0],
            [4.5, 2.0],
            [0.5, 3.5],
        ],
        dtype=np.float64,
    )
    inside = points_inside_boundary_loops_2d(pts, loops)
    assert [bool(v) for v in inside] == [True, False, False, True]


def test_boundary_loop_builder_rejects_branching_vertices():
    edges = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [2.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match='degree-2 loops'):
        validate_boundary_edges_2d(edges)
    with pytest.raises(ValueError, match='degree-2 loops'):
        build_boundary_loops_2d(edges)


def test_boundary_points_are_promoted_to_inside_consistently():
    edges = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    loops = build_boundary_loops_2d(edges)
    pts = np.asarray(
        [
            [0.5, 0.5],   # interior
            [1.0, 0.5],   # edge
            [0.0, 0.0],   # vertex
            [1.2, 0.5],   # exterior
        ],
        dtype=np.float64,
    )
    inside, on_boundary = points_inside_boundary_loops_2d_with_boundary(pts, loops, on_edge_tol=1.0e-9)
    assert [bool(v) for v in inside] == [True, True, True, False]
    assert [bool(v) for v in on_boundary] == [False, True, True, False]


def test_scalar_boundary_edge_inside_matches_loop_truth_for_holes_and_boundary():
    edges = np.asarray(
        [
            [[0.0, 0.0], [4.0, 0.0]],
            [[4.0, 0.0], [4.0, 4.0]],
            [[4.0, 4.0], [0.0, 4.0]],
            [[0.0, 4.0], [0.0, 0.0]],
            [[1.0, 1.0], [3.0, 1.0]],
            [[3.0, 1.0], [3.0, 3.0]],
            [[3.0, 3.0], [1.0, 3.0]],
            [[1.0, 3.0], [1.0, 1.0]],
        ],
        dtype=np.float64,
    )
    loops = build_boundary_loops_2d(edges)
    pts = np.asarray(
        [
            [0.5, 0.5],
            [2.0, 2.0],
            [4.5, 2.0],
            [0.5, 3.5],
            [4.0, 2.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )
    inside_vec, boundary_vec = points_inside_boundary_loops_2d_with_boundary(pts, loops, on_edge_tol=1.0e-9)
    scalar = [point_inside_boundary_edges_2d_with_boundary(pt, edges, on_edge_tol=1.0e-9) for pt in pts]
    assert [v[0] for v in scalar] == [bool(v) for v in inside_vec]
    assert [v[1] for v in scalar] == [bool(v) for v in boundary_vec]


def test_loop_bisection_fallback_returns_boundary_hit_for_crossing_segment():
    prepared = build_prepared_runtime_from_yaml(ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml')
    runtime = prepared.runtime
    hit = segment_hit_from_loop_bisection(
        runtime,
        p0=np.asarray([0.0, 0.0], dtype=np.float64),
        p1=np.asarray([2.0, 0.0], dtype=np.float64),
        on_boundary_tol_m=1.0e-7,
    )
    assert hit is not None
    assert isinstance(hit, BoundaryHit)
    assert 0.0 < float(hit.alpha_hint) < 1.0
    assert np.isclose(np.linalg.norm(hit.normal), 1.0, atol=1.0e-6)
    assert int(hit.part_id) >= 0


def test_precomputed_geometry_reads_new_and_legacy_boundary_part_keys(tmp_path: Path):
    mesh = parse_comsol_mphtxt(ROOT / 'data' / 'argon_gec_ccp_base2.mphtxt')
    arrays = build_precomputed_arrays(mesh, diagnostic_grid_spacing_m=1.0e-3)
    common = {
        'axis_0': arrays['axes_x'],
        'axis_1': arrays['axes_y'],
        'sdf': arrays['sdf'],
        'valid_mask': arrays['inside'].astype(bool),
        'normal_0': arrays['normal_x'],
        'normal_1': arrays['normal_y'],
        'boundary_edges': arrays['boundary_edges'],
        'boundary_edge_part_ids': arrays['boundary_part_ids'],
        'boundary_loops_2d_flat': arrays['boundary_loops_2d_flat'],
        'boundary_loops_2d_offsets': arrays['boundary_loops_2d_offsets'],
    }
    new_path = tmp_path / 'geom_new.npz'
    old_path = tmp_path / 'geom_old.npz'
    np.savez_compressed(new_path, nearest_boundary_part_id_map=arrays['nearest_boundary_part_id_map'], **common)
    np.savez_compressed(old_path, part_id_map=arrays['nearest_boundary_part_id_map'], **common)

    geom_new = build_precomputed_geometry({'npz_path': str(new_path)}, spatial_dim=2, coordinate_system='cartesian_xy')
    geom_old = build_precomputed_geometry({'npz_path': str(old_path)}, spatial_dim=2, coordinate_system='cartesian_xy')

    assert np.array_equal(geom_new.geometry.nearest_boundary_part_id_map, arrays['nearest_boundary_part_id_map'])
    assert np.array_equal(geom_old.geometry.nearest_boundary_part_id_map, arrays['nearest_boundary_part_id_map'])
    assert len(geom_new.geometry.boundary_loops_2d) >= 1
    assert geom_new.geometry.metadata['boundary_edge_topology']['branch_vertex_count'] == 0
    assert geom_new.geometry.metadata['boundary_edge_topology']['dangling_vertex_count'] == 0
    assert int(geom_new.geometry.metadata['boundary_loop_count_2d']) >= 1


def test_precomputed_geometry_3d_rejects_non_closed_surface(tmp_path: Path):
    tri = _cube_triangles_oriented()[:-1]
    axis = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    shape = (axis.size, axis.size, axis.size)
    npz_path = tmp_path / 'bad_geom_3d.npz'
    np.savez_compressed(
        npz_path,
        axis_0=axis,
        axis_1=axis,
        axis_2=axis,
        sdf=np.zeros(shape, dtype=np.float64),
        valid_mask=np.ones(shape, dtype=bool),
        nearest_boundary_part_id_map=np.ones(shape, dtype=np.int32),
        normal_0=np.zeros(shape, dtype=np.float64),
        normal_1=np.zeros(shape, dtype=np.float64),
        normal_2=np.ones(shape, dtype=np.float64),
        boundary_triangles=tri,
        boundary_triangle_part_ids=np.ones(tri.shape[0], dtype=np.int32),
    )
    with pytest.raises(ValueError, match='closed 2-manifold'):
        build_precomputed_geometry({'npz_path': str(npz_path)}, spatial_dim=3, coordinate_system='cartesian_xyz')


def test_sample_points_in_quads_stay_inside_actual_domain():
    mesh = parse_comsol_mphtxt(ROOT / 'data' / 'argon_gec_ccp_base2.mphtxt')
    pts = _sample_points_in_quads(mesh.vertices, mesh.type_blocks['quad'].elements, count=256, seed=7)
    inside = _points_inside_quads(mesh.vertices, mesh.type_blocks['quad'].elements, pts)
    assert pts.shape == (256, 2)
    assert bool(np.all(inside))


def test_visualization_unified_clean_and_index(tmp_path: Path):
    out_dir = tmp_path / 'out_visualization_index'
    run_solver_2d_from_yaml(ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml', output_dir=out_dir)

    # legacy dirs to be cleaned
    for legacy in ('graphs', 'animations', 'visuals'):
        d = out_dir / legacy
        d.mkdir(parents=True, exist_ok=True)
        (d / 'legacy.txt').write_text('old', encoding='utf-8')

    index_path = export_visualizations(
        output_dir=out_dir,
        case_dir=ROOT / 'examples' / 'minimal_2d',
        modules=('graphs', 'animations'),
        clean=True,
        sample_trajectories=16,
        animation_sample_count=16,
        animation_fps=3,
        animation_interpolate_factor=2,
        overlay_wall_events=True,
    )
    assert index_path.exists()
    for legacy in ('graphs', 'animations', 'visuals'):
        assert not (out_dir / legacy).exists()
    index = json.loads(index_path.read_text(encoding='utf-8'))
    assert set(index['modules'].keys()) == {'graphs', 'animations'}
    assert index['health_summary']['status'] in {'pass', 'review'}
    assert (out_dir / 'visualizations' / 'reports' / 'run_summary.md').exists()
    assert (out_dir / 'visualizations' / 'graphs' / 'graph_summary.json').exists()
    assert (out_dir / 'visualizations' / 'animations' / 'animation_report.json').exists()


def test_unified_visualizations_3d_projection_gifs(tmp_path: Path):
    out_dir = tmp_path / 'out_visualization_3d'
    run_solver_3d_from_yaml(ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml', output_dir=out_dir)
    index_path = export_visualizations(
        output_dir=out_dir,
        modules=('animations',),
        clean=True,
        animation_sample_count=24,
        animation_fps=3,
        animation_interpolate_factor=2,
        overlay_wall_events=True,
    )
    assert index_path.exists()
    anim_dir = out_dir / 'visualizations' / 'animations'
    assert (anim_dir / 'trajectories_all_particles_xy.gif').exists()
    assert (anim_dir / 'trajectories_all_particles_xz.gif').exists()
    assert (anim_dir / 'trajectories_all_particles_yz.gif').exists()


def test_collision_diagnostics_are_written_and_max_hits_limit_is_applied(tmp_path: Path):
    particles = pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': 0.0,
                'y': 0.0,
                'vx': 50.0,
                'vy': 0.0,
                'release_time': 0.0,
                'mass': 1e-15,
                'diameter': 1e-6,
                'density': 1200.0,
                'charge': 0.0,
                'source_part_id': 10,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    )
    particles_path = tmp_path / 'fast_particles.csv'
    particles.to_csv(particles_path, index=False)

    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: (
            cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())}),
            cfg.setdefault('solver', {}).update(
                    {
                        'dt': 0.2,
                        't_end': 0.2,
                        'save_every': 1,
                        'min_tau_p_s': 1.0,
                        'max_wall_hits_per_step': 1,
                        'min_remaining_dt_ratio': 0.0,
                    }
                ),
            cfg.setdefault('output', {}).update({'write_collision_diagnostics': 1}),
        ),
    )
    out_dir = tmp_path / 'out_diag_hits'
    run_solver_2d_from_yaml(config_path, output_dir=out_dir)

    diag_path = out_dir / 'collision_diagnostics.json'
    assert diag_path.exists()
    diag = json.loads(diag_path.read_text(encoding='utf-8'))
    assert int(diag['max_wall_hits_per_step']) == 1
    assert int(diag['max_hits_reached_count']) >= 1
    assert int(diag['multi_hit_events_count']) == 0
    assert int(diag['max_hit_event_summary']['event_count']) == int(diag['max_hits_reached_count'])
    assert float(diag['max_hit_event_summary']['remaining_dt_total_s']) >= 0.0
    max_hit_events = pd.read_csv(out_dir / 'max_hit_events.csv')
    assert not max_hit_events.empty
    assert set(max_hit_events.columns).issuperset({'time_s', 'particle_id', 'hits_in_step', 'remaining_dt_s', 'part_id_sequence'})
    assert int(max_hit_events['hits_in_step'].max()) >= 1


def test_collision_reintegration_counter_is_nonzero_for_2d_wall_bounces(tmp_path: Path):
    particles = pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': 0.0,
                'y': 0.0,
                'vx': 50.0,
                'vy': 0.0,
                'release_time': 0.0,
                'mass': 1e-15,
                'diameter': 1e-6,
                'density': 1200.0,
                'charge': 0.0,
                'source_part_id': 10,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    )
    particles_path = tmp_path / 'reintegrate_particles_2d.csv'
    particles.to_csv(particles_path, index=False)
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: (
            cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())}),
            cfg.setdefault('solver', {}).update(
                {'dt': 0.2, 't_end': 0.2, 'save_every': 1, 'min_tau_p_s': 1.0, 'max_wall_hits_per_step': 3, 'min_remaining_dt_ratio': 0.0}
            ),
            cfg.setdefault('output', {}).update({'write_collision_diagnostics': 1}),
        ),
    )
    out_dir = tmp_path / 'out_diag_reintegrate_2d'
    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    diag = json.loads((out_dir / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    assert int(diag['collision_reintegrated_segments_count']) >= 1


def test_collision_reintegration_counter_is_nonzero_for_3d_wall_bounces(tmp_path: Path):
    particles = pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': 0.0,
                'y': 0.0,
                'z': 0.0,
                'vx': 80.0,
                'vy': 0.0,
                'vz': 0.0,
                'release_time': 0.0,
                'mass': 1e-15,
                'diameter': 1e-6,
                'density': 1200.0,
                'charge': 0.0,
                'source_part_id': 10,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    )
    particles_path = tmp_path / 'reintegrate_particles_3d.csv'
    particles.to_csv(particles_path, index=False)
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml',
        mutate=lambda cfg: (
            cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())}),
            cfg.setdefault('solver', {}).update(
                {'dt': 0.2, 't_end': 0.2, 'save_every': 1, 'min_tau_p_s': 1.0, 'max_wall_hits_per_step': 3, 'min_remaining_dt_ratio': 0.0}
            ),
            cfg.setdefault('output', {}).update({'write_collision_diagnostics': 1}),
        ),
    )
    out_dir = tmp_path / 'out_diag_reintegrate_3d'
    run_solver_3d_from_yaml(config_path, output_dir=out_dir)
    diag = json.loads((out_dir / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    assert int(diag['collision_reintegrated_segments_count']) >= 1


def test_adaptive_substep_diagnostics_toggle(tmp_path: Path):
    particles = pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': 0.0,
                'y': 0.0,
                'vx': 5.0,
                'vy': 0.0,
                'release_time': 0.0,
                'mass': 1e-15,
                'diameter': 1e-5,
                'density': 1200.0,
                'charge': 0.0,
                'source_part_id': 10,
                'material_id': 1,
                'source_event_tag': '',
                'stick_probability': 0.0,
            }
        ]
    )
    particles_path = tmp_path / 'adaptive_particles_2d.csv'
    particles.to_csv(particles_path, index=False)
    base_mutation = lambda cfg: (
        cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())}),
        cfg.setdefault('providers', {}).setdefault('geometry', {}).update({'bounds': [-10.0, 10.0, -10.0, 10.0], 'grid_shape': [51, 51]}),
        cfg.setdefault('providers', {}).setdefault('field', {}).update({'shear_rate': 0.0}),
        cfg.setdefault('solver', {}).update({'dt': 0.1, 't_end': 0.2, 'save_every': 1, 'min_tau_p_s': 1e-8, 'integrator': 'etd'}),
        cfg.setdefault('output', {}).update({'write_collision_diagnostics': 1}),
    )

    adaptive_off_dir = tmp_path / 'adaptive_off'
    adaptive_on_dir = tmp_path / 'adaptive_on'
    adaptive_off_dir.mkdir(parents=True, exist_ok=True)
    adaptive_on_dir.mkdir(parents=True, exist_ok=True)

    cfg_off = _write_config(
        adaptive_off_dir,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: (
            base_mutation(cfg),
            cfg.setdefault('solver', {}).update({'adaptive_substep_enabled': 0, 'adaptive_substep_tau_ratio': 0.5, 'adaptive_substep_max_splits': 4}),
        ),
    )
    cfg_on = _write_config(
        adaptive_on_dir,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: (
            base_mutation(cfg),
            cfg.setdefault('solver', {}).update({'adaptive_substep_enabled': 1, 'adaptive_substep_tau_ratio': 0.5, 'adaptive_substep_max_splits': 4}),
        ),
    )
    out_off = tmp_path / 'out_adaptive_off'
    out_on = tmp_path / 'out_adaptive_on'
    run_solver_2d_from_yaml(cfg_off, output_dir=out_off)
    run_solver_2d_from_yaml(cfg_on, output_dir=out_on)
    diag_off = json.loads((out_off / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    diag_on = json.loads((out_on / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    assert int(diag_off['adaptive_substep_segments_count']) == 0
    assert int(diag_off['adaptive_substep_trigger_count']) == 0
    assert int(diag_on['adaptive_substep_enabled']) == 1
    assert int(diag_on['adaptive_substep_segments_count']) > 0
    assert int(diag_on['adaptive_substep_trigger_count']) > 0


def test_default_max_wall_hits_per_step_is_5(tmp_path: Path):
    out_dir = tmp_path / 'out_default_hits'
    config_path = _write_config(tmp_path, ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml')
    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    diag = json.loads((out_dir / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    assert int(diag['max_wall_hits_per_step']) == 5
    assert int(diag['adaptive_substep_segments_count']) == 0
    assert int(diag['adaptive_substep_trigger_count']) == 0


def test_animation_helpers_support_interpolated_wall_event_overlay():
    positions = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.5, 0.0], [1.0, 0.5]],
            [[1.0, 0.0], [1.0, 1.0]],
        ],
        dtype=np.float64,
    )
    times = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    positions_i, times_i = _interpolate_frames(positions, times, factor=3)
    assert positions_i.shape[0] == 7
    assert times_i.shape[0] == 7

    wall_events = pd.DataFrame(
        {
            'time_s': [0.25, 1.75, 0.5],
            'particle_id': [10, 10, 99],
        }
    )
    xy, frame_ids = _prepare_event_overlay(
        wall_events=wall_events,
        sample_indices=np.asarray([0], dtype=np.int64),
        particle_ids=np.asarray([10, 20], dtype=np.int64),
        positions=positions_i,
        times=times_i,
        interpolate_positions=True,
    )
    assert xy.shape == (2, 2)
    assert frame_ids.shape == (2,)
    assert int(np.min(frame_ids)) >= 0
    assert int(np.max(frame_ids)) < positions_i.shape[0]
