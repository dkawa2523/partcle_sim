from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
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
    points_inside_boundary_loops_2d,
    points_inside_boundary_loops_2d_with_boundary,
    validate_boundary_edges_2d,
)
from particle_tracer_unified.core.boundary_service import (
    BoundaryHit,
    build_boundary_service,
    normalize_polyline_alpha,
    polyline_hit_from_boundary_edges,
    segment_hit_from_loop_bisection,
)
from particle_tracer_unified.core.field_backend import field_backend_kind, sample_field_valid_status
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
from particle_tracer_unified.core.catalogs import resolve_step_wall_model
from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    GeometryND,
    GeometryProviderND,
    ProcessStepRow,
    ProcessStepTable,
    QuantitySeriesND,
    RegularFieldND,
    TriangleMeshField2D,
    WallCatalog,
    WallPartModel,
    with_process_step_explicit_fields,
)
from particle_tracer_unified.core.integrator_registry import get_integrator_spec, integrator_spec_from_mode
from particle_tracer_unified.core.process_steps import apply_process_step_controls
from particle_tracer_unified.core.source_registry import get_source_law
from particle_tracer_unified.core.source_resolution import global_source_defaults
from particle_tracer_unified.io.field_regularization import regularize_precomputed_field_to_geometry
from particle_tracer_unified.io.runtime_builder import build_prepared_runtime_from_yaml, build_runtime_from_config
from particle_tracer_unified.io.tables import (
    load_materials_csv,
    load_part_walls_csv,
    load_process_steps_csv,
    load_recipe_manifest_yaml,
)
from particle_tracer_unified.providers.precomputed import build_precomputed_geometry, build_precomputed_triangle_mesh_field
from particle_tracer_unified.solvers.high_fidelity_collision import (
    _apply_wall_hit_step,
    _classify_trial_collisions,
    _advance_colliding_particle,
)
from particle_tracer_unified.solvers.high_fidelity_freeflight import (
    RegularRectilinearCompiledBackend,
    TriangleMesh2DCompiledBackend,
    ValidMaskPrefixResolution,
    _compile_runtime_arrays,
)
from particle_tracer_unified.solvers.high_fidelity_runtime import (
    _apply_valid_mask_retry_then_stop,
    _initial_collision_diagnostics,
    run_prepared_runtime,
)
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
    write_case_files,
)
from tools.compare_against_reference import class_match_ratio, main as compare_against_reference_main
from tools.evaluate_valid_mask_rollout import main as evaluate_valid_mask_rollout_main
from tools.export_boundary_diagnostics_visuals import export_boundary_diagnostics
from tools.export_trajectory_animation import _interpolate_frames, _prepare_event_overlay
from tools.export_mechanics_visuals import export_mechanics_visuals
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


def test_process_steps_csv_source_output_fields_are_loaded():
    table = load_process_steps_csv(ROOT / 'schemas' / 'process_steps.example.csv')
    rows = {r.step_name: r for r in table.rows}
    assert rows['etch'].source_law_override == 'flake_burst_material'
    assert rows['etch'].source_speed_scale == pytest.approx(1.15)
    assert rows['etch'].source_event_gain_scale == pytest.approx(1.20)
    assert rows['purge'].output_write_wall_events == 0
    assert rows['afterglow'].output_save_positions == 0
    assert rows['afterglow'].output_write_diagnostics == 1


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
    assert compiled.extension_band_mask.shape == (3, 3)


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


def test_runtime_builder_regularizes_precomputed_field_with_geometry_narrow_band(tmp_path: Path):
    axes = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    valid_mask = np.asarray(
        [
            [1, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    )
    geom_path = _write_precomputed_geometry_npz(tmp_path / 'geom.npz', axes, axes, valid_mask=valid_mask)
    field_path = tmp_path / 'field.npz'
    payload = {
        'axis_0': axes,
        'axis_1': axes,
        'times': np.asarray([0.0], dtype=np.float64),
        'valid_mask': valid_mask,
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

    assert np.array_equal(np.asarray(field.core_valid_mask, dtype=bool), valid_mask)
    assert str(field.metadata['field_regularization_mode']) == 'geometry_narrow_band_normal_probe'
    assert int(np.count_nonzero(field.extension_band_mask)) == 5
    assert int(field.metadata['field_regularization_added_node_count']) == 5
    assert int(field.metadata['field_regularization_probe_success_count']) == 2
    assert int(field.metadata['field_regularization_probe_fallback_count']) == 3
    assert bool(field.extension_band_mask[0, 2]) is True
    assert bool(field.extension_band_mask[2, 0]) is True
    assert float(field.quantities['ux'].data[0, 2]) == pytest.approx(float(2.0 - np.sqrt(2.0)))
    assert float(field.quantities['ux'].data[2, 0]) == pytest.approx(10.0)
    status_before = sample_valid_mask_status(valid_mask, (axes, axes), np.asarray([1.75, 0.5], dtype=np.float64))
    status_after = sample_valid_mask_status(
        np.asarray(field.valid_mask, dtype=bool),
        field.axes,
        np.asarray([1.75, 0.5], dtype=np.float64),
    )
    assert int(status_before) == int(VALID_MASK_STATUS_HARD_INVALID)
    assert int(status_after) != int(VALID_MASK_STATUS_HARD_INVALID)


def test_recipe_manifest_source_output_fields_are_loaded():
    table = load_recipe_manifest_yaml(ROOT / 'schemas' / 'recipe_manifest.example.yaml')
    rows = {r.step_name: r for r in table.rows}
    assert rows['etch'].source_law_override == 'flake_burst_material'
    assert rows['purge'].source_speed_scale == pytest.approx(0.90)
    assert rows['purge'].output_write_wall_events == 0
    assert rows['afterglow'].output_save_positions == 0
    assert rows['afterglow'].output_write_diagnostics == 1


def test_step_defaults_can_explicitly_override_wall_coefficients_to_legacy_default_values():
    base_model = WallPartModel(
        part_id=7,
        part_name='wall_7',
        material_id=1,
        material_name='steel',
        law_name='stick',
        stick_probability=0.2,
        restitution=0.82,
        diffuse_fraction=0.35,
        critical_sticking_velocity_mps=0.1,
        reflectivity=0.0,
        roughness_rms=0.0,
        metadata={},
    )
    wall_catalog = WallCatalog(default_model=base_model, part_models=(base_model,), metadata={})
    step = ProcessStepRow(
        step_id=1,
        step_name='etch',
        start_s=0.0,
        end_s=1.0,
        metadata=with_process_step_explicit_fields({}, ()),
    )
    controlled = apply_process_step_controls(
        ProcessStepTable(rows=(step,), metadata={}),
        {
            'step_defaults': {
                'wall': {
                    'mode': 'mixed_specular_diffuse',
                    'restitution': 1.0,
                    'diffuse_fraction': 0.0,
                }
            }
        },
    )
    resolved = resolve_step_wall_model(wall_catalog, 7, controlled.rows[0])
    assert controlled.rows[0].wall_restitution == pytest.approx(1.0)
    assert controlled.rows[0].wall_diffuse_fraction == pytest.approx(0.0)
    assert resolved.law_name == 'mixed_specular_diffuse'
    assert resolved.restitution == pytest.approx(1.0)
    assert resolved.diffuse_fraction == pytest.approx(0.0)


def test_explicit_inherit_wall_mode_wins_over_process_default_wall_mode():
    base_model = WallPartModel(
        part_id=9,
        part_name='wall_9',
        material_id=1,
        material_name='steel',
        law_name='diffuse',
        stick_probability=0.1,
        restitution=0.75,
        diffuse_fraction=0.45,
        critical_sticking_velocity_mps=0.2,
        reflectivity=0.0,
        roughness_rms=0.0,
        metadata={},
    )
    wall_catalog = WallCatalog(default_model=base_model, part_models=(base_model,), metadata={})
    step = ProcessStepRow(
        step_id=1,
        step_name='etch',
        start_s=0.0,
        end_s=1.0,
        wall_mode='inherit',
        metadata=with_process_step_explicit_fields({}, ('wall_mode',)),
    )
    controlled = apply_process_step_controls(
        ProcessStepTable(rows=(step,), metadata={}),
        {'step_defaults': {'wall': {'mode': 'specular'}}},
    )
    resolved = resolve_step_wall_model(wall_catalog, 9, controlled.rows[0])
    assert controlled.rows[0].wall_mode == 'inherit'
    assert resolved.law_name == 'diffuse'


def test_runtime_builder_rejects_multiple_process_step_sources(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('paths', {}).update({'recipe_manifest_yaml': str((ROOT / 'schemas' / 'recipe_manifest.example.yaml').resolve())}),
    )
    with pytest.raises(ValueError, match='Specify only one of paths.process_steps_csv or paths.recipe_manifest_yaml'):
        build_prepared_runtime_from_yaml(config_path)


def test_runtime_builder_rejects_process_step_gaps(tmp_path: Path):
    steps = pd.DataFrame(
        [
            {'step_id': 1, 'step_name': 'etch', 'start_s': 0.0, 'end_s': 0.5},
            {'step_id': 2, 'step_name': 'purge', 'start_s': 0.75, 'end_s': 1.0},
        ]
    )
    steps_path = tmp_path / 'process_steps_gap.csv'
    steps.to_csv(steps_path, index=False)
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('paths', {}).update({'process_steps_csv': str(steps_path.resolve())}),
    )
    with pytest.raises(ValueError, match='Process steps contain a gap'):
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


def test_runtime_builder_applies_step_aware_source_preprocess_controls(tmp_path: Path):
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('paths', {}).update(
            {'process_steps_csv': str((ROOT / 'schemas' / 'process_steps.example.csv').resolve())}
        ),
    )
    prepared = build_prepared_runtime_from_yaml(config_path)
    result = prepared.source_preprocess
    assert result is not None
    assert prepared.runtime.process_steps is not None
    assert [row.step_name for row in prepared.runtime.process_steps.rows] == ['etch', 'purge', 'afterglow']
    assert result.source_model_summary['law_usage'] == {'flake_burst_material': 3}
    assert {row['step_name'] for row in result.diagnostics_rows} == {'etch'}
    assert {row['step_source_law_override'] for row in result.diagnostics_rows} == {'flake_burst_material'}
    assert all(float(row['step_source_event_gain_scale']) == pytest.approx(1.2) for row in result.diagnostics_rows)


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
    assert labels.tolist() == ['active', 'invalid_mask_stopped', 'stuck', 'escaped']
    assert classes['particle_class'].tolist() == ['active', 'invalid_mask_stopped', 'stuck', 'escaped']


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
    assert summary['runs'][0]['unresolved_crossing_count'] >= 0
    assert 'field_regularization_mode' in summary['runs'][0]
    assert 'field_regularization_probe_success_count' in summary['runs'][0]


def test_evaluate_valid_mask_rollout_cli_writes_summary(tmp_path: Path):
    output_root = tmp_path / 'valid_mask_rollout'
    rc = evaluate_valid_mask_rollout_main(
        [
            '--config',
            str(ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml'),
            '--reference-config',
            str(ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml'),
            '--output-root',
            str(output_root),
            '--max-runtime-increase-ratio',
            '1.0',
            '--min-class-match-ratio',
            '1.0',
        ]
    )
    assert rc == 0
    summary_files = sorted(output_root.glob('valid_mask_rollout_*/rollout_summary.json'))
    assert summary_files
    summary = json.loads(summary_files[-1].read_text(encoding='utf-8'))
    assert summary['rollout_recommendation'] == 'candidate_ready_for_default'
    assert summary['checks']['runtime_increase_ok'] is True
    assert summary['checks']['class_match_ratio_ok'] is True
    assert summary['diagnostic']['run'] == 'diagnostic'
    assert summary['retry_then_stop']['run'] == 'retry_then_stop'
    assert 'field_regularization_mode' in summary['diagnostic']
    assert 'field_regularization_probe_success_count' in summary['diagnostic']


def test_evaluate_valid_mask_rollout_without_class_threshold_stays_fail_closed(tmp_path: Path):
    output_root = tmp_path / 'valid_mask_rollout_no_threshold'
    rc = evaluate_valid_mask_rollout_main(
        [
            '--config',
            str(ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml'),
            '--reference-config',
            str(ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml'),
            '--output-root',
            str(output_root),
            '--max-runtime-increase-ratio',
            '1.0',
        ]
    )
    assert rc == 1
    summary_files = sorted(output_root.glob('valid_mask_rollout_*/rollout_summary.json'))
    assert summary_files
    summary = json.loads(summary_files[-1].read_text(encoding='utf-8'))
    assert summary['rollout_recommendation'] == 'keep_opt_in'
    assert summary['checks']['class_match_ratio_ok'] is False


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
    assert set(diag_2d).issuperset(
        {
            'etd2_polyline_checks_count',
            'etd2_midpoint_outside_count',
            'etd2_polyline_hit_count',
            'etd2_polyline_fallback_count',
        }
    )

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
    assert set(diag_3d).issuperset(
        {
            'etd2_polyline_checks_count',
            'etd2_midpoint_outside_count',
            'etd2_polyline_hit_count',
            'etd2_polyline_fallback_count',
        }
    )


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
            cfg.setdefault('solver', {}).update({'integrator': 'etd2', 't_end': 0.12, 'save_every': 1})
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
    assert str(masked_report['field_regularization_mode']) == 'geometry_narrow_band_normal_probe'
    assert int(masked_report['field_regularization_added_node_count']) >= 0
    assert int(masked_report['field_regularization_probe_success_count']) + int(
        masked_report['field_regularization_probe_fallback_count']
    ) == int(masked_report['field_regularization_added_node_count'])
    assert int(masked_report['invalid_mask_stopped_count']) == 0
    assert 'valid_mask_violation_count_step' in masked_steps.columns
    assert 'valid_mask_mixed_stencil_count_step' in masked_steps.columns
    assert 'valid_mask_hard_invalid_count_step' in masked_steps.columns
    assert 'extension_band_sample_count_step' in masked_steps.columns
    assert 'invalid_mask_stopped_count_step' in masked_steps.columns
    assert int(masked_steps['valid_mask_violation_count_step'].sum()) > 0
    assert int(masked_steps['valid_mask_violation_count_step'].sum()) == int(
        masked_steps['valid_mask_mixed_stencil_count_step'].sum() + masked_steps['valid_mask_hard_invalid_count_step'].sum()
    )
    assert int(masked_steps['invalid_mask_stopped_count_step'].sum()) == 0


def test_geometry_narrow_band_regularization_uses_inward_normal_donor():
    axes = (np.arange(5, dtype=np.float64), np.arange(5, dtype=np.float64))
    xx, yy = np.meshgrid(axes[0], axes[1], indexing='ij')
    valid_mask = (xx + yy) <= 3.0
    sdf = (xx + yy - 3.0) / np.sqrt(2.0)
    normal_value = 1.0 / np.sqrt(2.0)
    normals = (
        np.full_like(xx, normal_value, dtype=np.float64),
        np.full_like(yy, normal_value, dtype=np.float64),
    )
    field_provider = _regular_field_provider_from_arrays(
        axes,
        valid_mask,
        quantities={'ux': (xx + yy).astype(np.float64)},
    )
    geometry_provider = _geometry_provider_from_arrays(axes, valid_mask, sdf, normals)

    regularized = regularize_precomputed_field_to_geometry(field_provider, geometry_provider)
    field = regularized.field

    assert str(field.metadata['field_regularization_mode']) == 'geometry_narrow_band_normal_probe'
    assert bool(field.extension_band_mask[2, 2]) is True
    assert int(field.metadata['field_regularization_probe_success_count']) > 0
    assert int(field.metadata['field_regularization_probe_success_count']) + int(
        field.metadata['field_regularization_probe_fallback_count']
    ) == int(field.metadata['field_regularization_added_node_count'])
    assert float(field.quantities['ux'].data[2, 2]) == pytest.approx(2.0)
    assert float(field.quantities['ux'].data[2, 2]) != pytest.approx(3.0)

    raw_status = sample_valid_mask_status(valid_mask, axes, np.asarray([1.9, 1.9], dtype=np.float64))
    regularized_status = sample_valid_mask_status(field.valid_mask, axes, np.asarray([1.9, 1.9], dtype=np.float64))
    assert int(raw_status) == int(VALID_MASK_STATUS_HARD_INVALID)
    assert int(regularized_status) != int(VALID_MASK_STATUS_HARD_INVALID)


def test_geometry_narrow_band_regularization_uses_nearest_core_fallback_when_probe_fails():
    axes = (np.asarray([0.0, 1.0, 2.0], dtype=np.float64), np.asarray([0.0, 1.0, 2.0], dtype=np.float64))
    valid_mask = np.zeros((3, 3), dtype=bool)
    valid_mask[1, 1] = True
    sdf = np.asarray(
        [
            [1.0, 0.8, 1.0],
            [0.8, -0.1, 0.8],
            [1.0, 0.8, 1.0],
        ],
        dtype=np.float64,
    )
    zero_normals = (
        np.zeros((3, 3), dtype=np.float64),
        np.zeros((3, 3), dtype=np.float64),
    )
    quantity = np.zeros((3, 3), dtype=np.float64)
    quantity[1, 1] = 42.0
    field_provider = _regular_field_provider_from_arrays(axes, valid_mask, quantities={'ux': quantity})
    geometry_provider = _geometry_provider_from_arrays(axes, valid_mask, sdf, zero_normals)

    regularized = regularize_precomputed_field_to_geometry(field_provider, geometry_provider)
    field = regularized.field

    assert int(field.metadata['field_regularization_added_node_count']) > 0
    assert int(field.metadata['field_regularization_probe_success_count']) == 0
    assert int(field.metadata['field_regularization_probe_fallback_count']) == int(
        field.metadata['field_regularization_added_node_count']
    )
    assert field.quantities['ux'].data[field.extension_band_mask] == pytest.approx(
        np.full(int(np.count_nonzero(field.extension_band_mask)), 42.0, dtype=np.float64)
    )


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
    )

    stopped_count = _apply_valid_mask_retry_then_stop(
        state=state,
        options=options,
        compiled={},
        spatial_dim=2,
        integrator_mode=0,
        dt_step=0.08,
        t_end_step=0.08,
        phys={'flow_scale': 1.0, 'drag_tau_scale': 1.0, 'body_accel_scale': 1.0, 'min_tau_p_s': 1.0e-9},
        body_accel=np.zeros(2, dtype=np.float64),
        tau_p=np.asarray([1.0], dtype=np.float64),
        flow_scale_particle=np.asarray([1.0], dtype=np.float64),
        drag_scale_particle=np.asarray([1.0], dtype=np.float64),
        body_scale_particle=np.asarray([1.0], dtype=np.float64),
    )

    assert int(stopped_count) == 0
    assert bool(state.invalid_mask_stopped[0]) is False
    assert int(state.collision_diagnostics['invalid_mask_retry_count']) == 0
    assert int(state.collision_diagnostics['invalid_mask_stopped_count']) == 0


def test_valid_mask_retry_then_stop_accepts_regularized_support_band(tmp_path: Path):
    axes = np.linspace(-1.0, 1.0, 81)
    field_path = tmp_path / 'field_extension_band.npz'
    _write_field_bundle(field_path, axes, axes)
    payload = {key: value for key, value in np.load(field_path).items()}
    payload['ux'] = np.zeros((axes.size, axes.size), dtype=np.float64)
    payload['uy'] = np.zeros((axes.size, axes.size), dtype=np.float64)
    valid_mask = np.ones((axes.size, axes.size), dtype=bool)
    valid_mask[-1, :] = False
    payload['valid_mask'] = valid_mask
    np.savez_compressed(field_path, **payload)

    particles_path = tmp_path / 'particles_extension_band.csv'
    pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': 0.995,
                'y': 0.0,
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

    cfg_dir = tmp_path / 'cfg_extension_band'
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
                    't_end': 0.02,
                    'dt': 0.02,
                    'save_every': 1,
                    'integrator': 'etd2',
                    'adaptive_substep_max_splits': 4,
                    'valid_mask_policy': 'retry_then_stop',
                }
            ),
        ),
    )

    out_dir = tmp_path / 'out_extension_band'
    run_solver_2d_from_yaml(cfg, output_dir=out_dir)

    final_df = pd.read_csv(out_dir / 'final_particles.csv')
    report = json.loads((out_dir / 'solver_report.json').read_text(encoding='utf-8'))
    diag = json.loads((out_dir / 'collision_diagnostics.json').read_text(encoding='utf-8'))

    assert int(final_df.loc[0, 'invalid_mask_stopped']) == 0
    assert int(report['invalid_mask_stopped_count']) == 0
    assert int(report['field_regularization_added_node_count']) > 0
    assert str(report['field_regularization_mode']) == 'geometry_narrow_band_normal_probe'
    assert int(report['field_regularization_probe_success_count']) + int(
        report['field_regularization_probe_fallback_count']
    ) == int(report['field_regularization_added_node_count'])
    assert int(diag['extension_band_sample_count']) > 0


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
                        'valid_mask_policy': 'retry_then_stop',
                    }
                ),
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
            np.asarray([-0.999, 0.0], dtype=np.float64),
            np.asarray([4.0, 0.0], dtype=np.float64),
            0.55,
            1,
            1,
            0,
            False,
        )[1:],
    )
    monkeypatch.setattr(
        'particle_tracer_unified.solvers.high_fidelity_collision.advance_freeflight_segment',
        lambda **kwargs: (
            np.asarray([0.8, 0.0], dtype=np.float64),
            np.asarray([4.0, 0.0], dtype=np.float64),
            1,
            np.asarray([[0.2, 0.0]], dtype=np.float64),
            VALID_MASK_STATUS_HARD_INVALID,
            False,
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
        segment_adaptive_enabled_for_retry=lambda _retry: 0,
        tau_p_i=1.0,
        flow_scale_particle_i=1.0,
        drag_scale_particle_i=1.0,
        body_scale_particle_i=1.0,
        global_flow_scale=1.0,
        global_drag_tau_scale=1.0,
        global_body_accel_scale=1.0,
        body_accel=np.zeros(2, dtype=np.float64),
        min_tau_p_s=1.0e-9,
        valid_mask_retry_then_stop_enabled=True,
        initial_x_next=np.asarray([-3.2, 0.0], dtype=np.float64),
        initial_v_next=np.asarray([-4.0, 0.0], dtype=np.float64),
        initial_stage_points=np.asarray([[-3.2, 0.0]], dtype=np.float64),
        initial_valid_mask_status=VALID_MASK_STATUS_CLEAN,
        initial_extension_band_sampled=False,
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
        wall_law_counts=wall_law_counts,
        wall_summary_counts=wall_summary_counts,
        stuck=stuck,
        absorbed=absorbed,
        active=active,
        max_wall_hits_per_step=5,
        max_hits_retry_splits=0,
        epsilon_offset_m=1.0e-6,
        on_boundary_tol_m=1.0e-6,
        triangle_surface_3d=None,
    )

    assert bool(result.invalid_mask_stopped) is True
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
            np.asarray([-0.999, 0.0], dtype=np.float64),
            np.asarray([4.0, 0.0], dtype=np.float64),
            0.55,
            1,
            1,
            0,
            False,
        )[1:],
    )
    monkeypatch.setattr(
        'particle_tracer_unified.solvers.high_fidelity_collision.advance_freeflight_segment',
        lambda **kwargs: (
            np.asarray([-0.7, 0.0], dtype=np.float64),
            np.asarray([4.0, 0.0], dtype=np.float64),
            1,
            np.asarray([[-0.7, 0.0]], dtype=np.float64),
            VALID_MASK_STATUS_HARD_INVALID,
            False,
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
        segment_adaptive_enabled_for_retry=lambda _retry: 0,
        tau_p_i=1.0,
        flow_scale_particle_i=1.0,
        drag_scale_particle_i=1.0,
        body_scale_particle_i=1.0,
        global_flow_scale=1.0,
        global_drag_tau_scale=1.0,
        global_body_accel_scale=1.0,
        body_accel=np.zeros(2, dtype=np.float64),
        min_tau_p_s=1.0e-9,
        valid_mask_retry_then_stop_enabled=True,
        initial_x_next=np.asarray([-3.2, 0.0], dtype=np.float64),
        initial_v_next=np.asarray([-4.0, 0.0], dtype=np.float64),
        initial_stage_points=np.asarray([[-3.2, 0.0]], dtype=np.float64),
        initial_valid_mask_status=VALID_MASK_STATUS_CLEAN,
        initial_extension_band_sampled=False,
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
        wall_law_counts={},
        wall_summary_counts={},
        stuck=np.zeros(1, dtype=bool),
        absorbed=np.zeros(1, dtype=bool),
        active=np.ones(1, dtype=bool),
        max_wall_hits_per_step=5,
        max_hits_retry_splits=0,
        epsilon_offset_m=1.0e-6,
        on_boundary_tol_m=1.0e-6,
        triangle_surface_3d=None,
    )

    assert bool(result.invalid_mask_stopped) is True
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
            np.asarray([-0.999, 0.0], dtype=np.float64),
            np.asarray([4.0, 0.0], dtype=np.float64),
            0.55,
            1,
            1,
            0,
            False,
        )[1:],
    )
    monkeypatch.setattr(
        'particle_tracer_unified.solvers.high_fidelity_collision.advance_freeflight_segment',
        lambda **kwargs: (
            np.asarray([-0.7, 0.0], dtype=np.float64),
            np.asarray([4.0, 0.0], dtype=np.float64),
            1,
            np.asarray([[-0.7, 0.0]], dtype=np.float64),
            VALID_MASK_STATUS_MIXED_STENCIL,
            False,
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
        segment_adaptive_enabled_for_retry=lambda _retry: 0,
        tau_p_i=1.0,
        flow_scale_particle_i=1.0,
        drag_scale_particle_i=1.0,
        body_scale_particle_i=1.0,
        global_flow_scale=1.0,
        global_drag_tau_scale=1.0,
        global_body_accel_scale=1.0,
        body_accel=np.zeros(2, dtype=np.float64),
        min_tau_p_s=1.0e-9,
        valid_mask_retry_then_stop_enabled=True,
        initial_x_next=np.asarray([-3.2, 0.0], dtype=np.float64),
        initial_v_next=np.asarray([-4.0, 0.0], dtype=np.float64),
        initial_stage_points=np.asarray([[-3.2, 0.0]], dtype=np.float64),
        initial_valid_mask_status=VALID_MASK_STATUS_CLEAN,
        initial_extension_band_sampled=False,
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
        wall_law_counts={},
        wall_summary_counts={},
        stuck=np.zeros(1, dtype=bool),
        absorbed=np.zeros(1, dtype=bool),
        active=np.ones(1, dtype=bool),
        max_wall_hits_per_step=5,
        max_hits_retry_splits=0,
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

    assert labels.tolist() == ['active', 'invalid_mask_stopped', 'stuck']
    assert counts['invalid_mask_stopped'] == 1
    assert step_state_count_series(step_df, 'invalid_mask_stopped').tolist() == pytest.approx([0.0, 1.0, 1.0])


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
            cfg.setdefault('process', {}).setdefault('step_defaults', {}).setdefault('wall', {}).update(
                {'mode': 'specular', 'stick_probability_scale': 0.0, 'restitution': 1.0, 'diffuse_fraction': 0.0}
            ),
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
            cfg.setdefault('process', {}).setdefault('step_defaults', {}).setdefault('wall', {}).update(
                {'mode': 'specular', 'stick_probability_scale': 0.0, 'restitution': 1.0, 'diffuse_fraction': 0.0}
            ),
            cfg.setdefault('process', {}).setdefault('step_overrides', {}).setdefault('etch', {}).setdefault('wall', {}).update(
                {'mode': 'specular', 'stick_probability_scale': 0.0, 'restitution': 1.0, 'diffuse_fraction': 0.0}
            ),
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
    collision_diagnostics = {
        'max_hits_retry_count': 0,
        'max_hits_reached_count': 0,
        'max_hits_retry_exhausted_count': 0,
        'dropped_remaining_dt_total_s': 0.0,
    }
    max_hit_rows: list[dict[str, object]] = []
    wall_rows: list[dict[str, object]] = []
    wall_law_counts: dict[str, int] = {}
    wall_summary_counts: dict[tuple[int, str, str], int] = {}
    stuck = np.asarray([False], dtype=bool)
    absorbed = np.asarray([False], dtype=bool)
    active = np.asarray([True], dtype=bool)

    x1, v1, rem1, hit_count1, total1, retry1, should_break1 = _apply_wall_hit_step(
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
        retry_splits_used=0,
        collision_diagnostics=collision_diagnostics,
        max_hit_rows=max_hit_rows,
        wall_rows=wall_rows,
        wall_law_counts=wall_law_counts,
        wall_summary_counts=wall_summary_counts,
        stuck=stuck,
        absorbed=absorbed,
        active=active,
        max_wall_hits_per_step=5,
        max_hits_retry_splits=0,
        min_remaining_dt=0.0,
        epsilon_offset_m=1.0e-6,
        on_boundary_tol_m=1.0e-6,
        t=0.2,
        triangle_surface_3d=None,
    )
    assert should_break1 is False
    assert rem1 == pytest.approx(0.18, abs=1e-15)

    _x2, _v2, rem2, _hit_count2, _total2, _retry2, should_break2 = _apply_wall_hit_step(
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
        remaining_dt=rem1,
        segment_dt=rem1,
        hit_count=hit_count1,
        total_hit_count=total1,
        hit_part_ids=[0],
        hit_outcomes=['reflected_specular'],
        retry_splits_used=retry1,
        collision_diagnostics=collision_diagnostics,
        max_hit_rows=max_hit_rows,
        wall_rows=wall_rows,
        wall_law_counts=wall_law_counts,
        wall_summary_counts=wall_summary_counts,
        stuck=stuck,
        absorbed=absorbed,
        active=active,
        max_wall_hits_per_step=5,
        max_hits_retry_splits=0,
        min_remaining_dt=0.0,
        epsilon_offset_m=1.0e-6,
        on_boundary_tol_m=1.0e-6,
        t=0.2,
        triangle_surface_3d=None,
    )
    assert should_break2 is False
    assert rem2 == pytest.approx(0.16, abs=1e-15)


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


def test_comsol_precomputed_case_runs(tmp_path: Path):
    out_dir = tmp_path / 'out_comsol_2d'
    cfg = ROOT / 'examples' / 'comsol_from_data_2d' / 'run_config.yaml'
    run_solver_2d_from_yaml(cfg, output_dir=out_dir)
    assert (out_dir / 'final_particles.csv').exists()
    wall_events = pd.read_csv(out_dir / 'wall_events.csv')
    assert not wall_events.empty
    assert int(wall_events['part_id'].min()) >= 1


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


def test_comsol_builder_writes_triangle_mesh_field_and_mesh_run_config(tmp_path: Path):
    out_dir = tmp_path / 'comsol_case_mesh'
    write_case_files(
        ROOT / 'data' / 'argon_gec_ccp_base2.mphtxt',
        out_dir,
        field_bundle_path=ROOT / 'data' / 'regridded_repo_field_bundle_argon_gec_ccp_base2_2d.npz',
        diagnostic_grid_spacing_m=5.0e-4,
    )
    mesh_field_npz = out_dir / 'generated' / 'comsol_field_mesh_2d.npz'
    mesh_run_config = out_dir / 'run_config_mesh.yaml'
    assert mesh_field_npz.exists()
    assert mesh_run_config.exists()
    with np.load(mesh_field_npz) as payload:
        assert 'mesh_vertices' in payload
        assert 'mesh_triangles' in payload
        assert 'ux' in payload and 'uy' in payload


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


def test_stuck_particles_snap_near_boundary_edges(tmp_path: Path):
    out_dir = tmp_path / 'out_comsol_stick'
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'comsol_from_data_2d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('solver', {}).update({'t_end': 0.4, 'save_every': 1, 'plot_particle_limit': 8}),
    )

    walls = pd.read_csv(ROOT / 'examples' / 'comsol_from_data_2d' / 'part_walls.csv')
    walls['wall_law'] = 'stick'
    walls['wall_stick_probability'] = 1.0
    walls_path = tmp_path / 'part_walls_stick.csv'
    walls.to_csv(walls_path, index=False)

    payload = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    providers = payload.setdefault('providers', {})
    geom_cfg = providers.setdefault('geometry', {})
    field_cfg = providers.setdefault('field', {})
    geom_cfg['npz_path'] = str((ROOT / 'examples' / 'comsol_from_data_2d' / 'generated' / 'comsol_geometry_2d.npz').resolve())
    field_cfg['npz_path'] = str((ROOT / 'examples' / 'comsol_from_data_2d' / 'generated' / 'comsol_field_2d.npz').resolve())
    payload.setdefault('paths', {})['part_walls_csv'] = str(walls_path.resolve())
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')

    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    final_df = pd.read_csv(out_dir / 'final_particles.csv')
    stuck = final_df[final_df['stuck'] == 1][['x', 'y']].to_numpy(dtype=np.float64)
    assert stuck.shape[0] > 0

    geom = np.load(ROOT / 'examples' / 'comsol_from_data_2d' / 'generated' / 'comsol_geometry_2d.npz')
    edges = np.asarray(geom['boundary_edges'], dtype=np.float64)
    p0 = edges[:, 0, :]
    p1 = edges[:, 1, :]
    ab = p1 - p0
    ab2 = np.sum(ab * ab, axis=1)
    mask = ab2 > 1e-30

    def _min_distance(point: np.ndarray) -> float:
        ap = point[None, :] - p0
        t = np.zeros(edges.shape[0], dtype=np.float64)
        t[mask] = np.clip(np.sum(ap[mask] * ab[mask], axis=1) / ab2[mask], 0.0, 1.0)
        proj = p0 + t[:, None] * ab
        d = np.linalg.norm(proj - point[None, :], axis=1)
        return float(np.min(d))

    dists = np.asarray([_min_distance(pt) for pt in stuck], dtype=np.float64)
    assert float(np.quantile(dists, 0.95)) < 1e-4


def test_loop_truth_keeps_non_escaped_particles_inside_geometry(tmp_path: Path):
    out_dir = tmp_path / 'out_comsol_loop_truth'
    report, prepared = run_solver_2d_from_yaml(ROOT / 'examples' / 'comsol_from_data_2d' / 'run_config.yaml', output_dir=out_dir)
    final_df = pd.read_csv(out_dir / 'final_particles.csv')
    loops = prepared.runtime.geometry_provider.geometry.boundary_loops_2d
    pts = final_df.loc[final_df['escaped'] == 0, ['x', 'y']].to_numpy(dtype=np.float64)
    inside = points_inside_boundary_loops_2d(pts, loops)
    assert bool(np.all(inside))


def test_wall_summary_is_written_even_when_wall_events_are_disabled(tmp_path: Path):
    out_dir = tmp_path / 'out_comsol_wall_summary'
    config_path = _write_config(tmp_path, ROOT / 'examples' / 'comsol_from_data_2d' / 'run_config.yaml')

    walls = pd.read_csv(ROOT / 'examples' / 'comsol_from_data_2d' / 'part_walls.csv')
    walls['wall_law'] = 'stick'
    walls['wall_stick_probability'] = 1.0
    walls_path = tmp_path / 'part_walls_stick.csv'
    walls.to_csv(walls_path, index=False)

    steps = pd.read_csv(ROOT / 'examples' / 'comsol_from_data_2d' / 'process_steps.csv')
    steps['output_write_wall_events'] = 0
    steps_path = tmp_path / 'process_steps_no_wall_events.csv'
    steps.to_csv(steps_path, index=False)

    payload = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    payload.setdefault('paths', {})['part_walls_csv'] = str(walls_path.resolve())
    payload.setdefault('paths', {})['process_steps_csv'] = str(steps_path.resolve())
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')

    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    wall_events = pd.read_csv(out_dir / 'wall_events.csv')
    wall_summary = pd.read_csv(out_dir / 'wall_summary_by_part.csv')
    assert wall_events.empty
    assert not wall_summary.empty
    assert int(wall_summary['count'].sum()) > 0
    wall_summary_json = yaml.safe_load((out_dir / 'wall_summary.json').read_text(encoding='utf-8'))
    assert int(wall_summary_json['total_wall_interactions']) == int(wall_summary['count'].sum())


def test_mechanics_csv_is_inside_only_and_graphs_use_wall_summary_fallback(tmp_path: Path):
    out_dir = tmp_path / 'out_comsol_visuals'
    config_path = _write_config(tmp_path, ROOT / 'examples' / 'comsol_from_data_2d' / 'run_config.yaml')

    walls = pd.read_csv(ROOT / 'examples' / 'comsol_from_data_2d' / 'part_walls.csv')
    walls['wall_law'] = 'stick'
    walls['wall_stick_probability'] = 1.0
    walls_path = tmp_path / 'part_walls_stick.csv'
    walls.to_csv(walls_path, index=False)

    steps = pd.read_csv(ROOT / 'examples' / 'comsol_from_data_2d' / 'process_steps.csv')
    steps['output_write_wall_events'] = 0
    steps_path = tmp_path / 'process_steps_no_wall_events.csv'
    steps.to_csv(steps_path, index=False)

    payload = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    payload.setdefault('paths', {})['part_walls_csv'] = str(walls_path.resolve())
    payload.setdefault('paths', {})['process_steps_csv'] = str(steps_path.resolve())
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding='utf-8')

    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    export_mechanics_visuals(ROOT / 'examples' / 'comsol_from_data_2d', out_dir, sample_trajectories=16, quiver_stride=16)
    export_result_graphs(out_dir, case_dir=ROOT / 'examples' / 'comsol_from_data_2d', sample_trajectories=16)

    mech = pd.read_csv(out_dir / 'visualizations' / 'mechanics' / 'mechanics_distribution_on_geometry.csv')
    assert not mech.empty
    assert set(mech.columns).issuperset({'inside', 'nearest_boundary_part_id'})
    assert set(int(v) for v in mech['inside'].unique()) == {1}
    assert (out_dir / 'visualizations' / 'graphs' / '08_wall_interactions_by_part_outcome.png').exists()
    assert (out_dir / 'visualizations' / 'graphs' / '09_stuck_counts_by_boundary_part.png').exists()


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
    assert int(diag['max_hits_retry_splits']) == 0
    assert int(diag['max_hits_retry_local_adaptive_enabled']) == 0
    assert int(diag['max_hits_reached_count']) >= 1
    assert int(diag['multi_hit_events_count']) == 0
    assert int(diag['max_hits_retry_count']) == 0
    assert int(diag['max_hits_retry_exhausted_count']) == 0
    assert float(diag['dropped_remaining_dt_total_s']) == pytest.approx(0.0, abs=1e-15)
    max_hit_events = pd.read_csv(out_dir / 'max_hit_events.csv')
    assert not max_hit_events.empty
    assert set(max_hit_events.columns).issuperset({'time_s', 'particle_id', 'hits_in_step', 'remaining_dt_s', 'part_id_sequence'})
    assert int(max_hit_events['hits_in_step'].max()) >= 1


def test_max_hits_retry_splits_improves_or_matches_2d_max_hit_drop_metrics(tmp_path: Path):
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
    particles_path = tmp_path / 'retry_particles_2d.csv'
    particles.to_csv(particles_path, index=False)

    def _mutate(cfg, retry_splits: int):
        cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())})
        cfg.setdefault('solver', {}).update(
            {
                'dt': 0.2,
                't_end': 0.2,
                'save_every': 1,
                'min_tau_p_s': 1.0,
                'max_wall_hits_per_step': 1,
                'min_remaining_dt_ratio': 0.0,
                'max_hits_retry_splits': int(retry_splits),
            }
        )
        cfg.setdefault('process', {}).setdefault('step_defaults', {}).setdefault('wall', {}).update(
            {'mode': 'specular', 'stick_probability_scale': 0.0, 'restitution': 1.0, 'diffuse_fraction': 0.0}
        )
        cfg.setdefault('output', {}).update({'write_collision_diagnostics': 1})

    cfg_off_dir = tmp_path / 'cfg_retry_off_2d'
    cfg_on_dir = tmp_path / 'cfg_retry_on_2d'
    cfg_off_dir.mkdir(parents=True, exist_ok=True)
    cfg_on_dir.mkdir(parents=True, exist_ok=True)
    cfg_off = _write_config(cfg_off_dir, ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml', mutate=lambda cfg: _mutate(cfg, 0))
    cfg_on = _write_config(cfg_on_dir, ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml', mutate=lambda cfg: _mutate(cfg, 2))

    out_off = tmp_path / 'out_retry_off_2d'
    out_on = tmp_path / 'out_retry_on_2d'
    run_solver_2d_from_yaml(cfg_off, output_dir=out_off)
    run_solver_2d_from_yaml(cfg_on, output_dir=out_on)

    diag_off = json.loads((out_off / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    diag_on = json.loads((out_on / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    assert int(diag_off['max_hits_retry_splits']) == 0
    assert int(diag_on['max_hits_retry_splits']) == 2
    assert int(diag_on['max_hits_retry_count']) <= 2
    assert int(diag_on['max_hits_reached_count']) <= int(diag_off['max_hits_reached_count'])
    assert int(diag_off['max_hits_retry_exhausted_count']) == 0
    assert float(diag_off['dropped_remaining_dt_total_s']) == pytest.approx(0.0, abs=1e-15)
    assert int(diag_on['max_hits_retry_exhausted_count']) <= int(diag_on['max_hits_reached_count'])
    assert float(diag_on['dropped_remaining_dt_total_s']) >= 0.0


def test_max_hits_retry_local_adaptive_improves_or_matches_2d_max_hit_drop_metrics(tmp_path: Path):
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
    particles_path = tmp_path / 'retry_local_adaptive_particles_2d.csv'
    particles.to_csv(particles_path, index=False)

    def _mutate(cfg, local_adaptive_enabled: int):
        cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())})
        cfg.setdefault('solver', {}).update(
            {
                'dt': 0.2,
                't_end': 0.2,
                'save_every': 1,
                'min_tau_p_s': 1.0,
                'max_wall_hits_per_step': 1,
                'min_remaining_dt_ratio': 0.0,
                'max_hits_retry_splits': 2,
                'adaptive_substep_enabled': 0,
                'max_hits_retry_local_adaptive_enabled': int(local_adaptive_enabled),
                'adaptive_substep_tau_ratio': 0.01,
                'adaptive_substep_max_splits': 3,
            }
        )
        cfg.setdefault('process', {}).setdefault('step_defaults', {}).setdefault('wall', {}).update(
            {'mode': 'specular', 'stick_probability_scale': 0.0, 'restitution': 1.0, 'diffuse_fraction': 0.0}
        )
        cfg.setdefault('output', {}).update({'write_collision_diagnostics': 1})

    cfg_off_dir = tmp_path / 'cfg_retry_local_adaptive_off_2d'
    cfg_on_dir = tmp_path / 'cfg_retry_local_adaptive_on_2d'
    cfg_off_dir.mkdir(parents=True, exist_ok=True)
    cfg_on_dir.mkdir(parents=True, exist_ok=True)
    cfg_off = _write_config(cfg_off_dir, ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml', mutate=lambda cfg: _mutate(cfg, 0))
    cfg_on = _write_config(cfg_on_dir, ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml', mutate=lambda cfg: _mutate(cfg, 1))

    out_off = tmp_path / 'out_retry_local_adaptive_off_2d'
    out_on = tmp_path / 'out_retry_local_adaptive_on_2d'
    run_solver_2d_from_yaml(cfg_off, output_dir=out_off)
    run_solver_2d_from_yaml(cfg_on, output_dir=out_on)

    diag_off = json.loads((out_off / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    diag_on = json.loads((out_on / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    assert int(diag_off['adaptive_substep_enabled']) == 0
    assert int(diag_on['adaptive_substep_enabled']) == 0
    assert int(diag_off['max_hits_retry_local_adaptive_enabled']) == 0
    assert int(diag_on['max_hits_retry_local_adaptive_enabled']) == 1
    assert int(diag_on['max_hits_reached_count']) <= int(diag_off['max_hits_reached_count'])
    assert int(diag_on['unresolved_crossing_count']) == 0
    assert int(diag_on['max_hits_retry_count']) >= 1
    assert int(diag_off['adaptive_substep_segments_count']) == 0
    assert int(diag_off['adaptive_substep_trigger_count']) == 0
    assert int(diag_on['adaptive_substep_segments_count']) > 0
    assert int(diag_on['adaptive_substep_trigger_count']) > 0


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
            cfg.setdefault('process', {}).setdefault('step_defaults', {}).setdefault('wall', {}).update(
                {'mode': 'specular', 'stick_probability_scale': 0.0, 'restitution': 1.0, 'diffuse_fraction': 0.0}
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
            cfg.setdefault('process', {}).setdefault('step_defaults', {}).setdefault('wall', {}).update(
                {'mode': 'specular', 'stick_probability_scale': 0.0, 'restitution': 1.0, 'diffuse_fraction': 0.0}
            ),
            cfg.setdefault('output', {}).update({'write_collision_diagnostics': 1}),
        ),
    )
    out_dir = tmp_path / 'out_diag_reintegrate_3d'
    run_solver_3d_from_yaml(config_path, output_dir=out_dir)
    diag = json.loads((out_dir / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    assert int(diag['collision_reintegrated_segments_count']) >= 1


def test_max_hits_retry_splits_improves_or_matches_3d_max_hit_drop_metrics(tmp_path: Path):
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
    particles_path = tmp_path / 'retry_particles_3d.csv'
    particles.to_csv(particles_path, index=False)

    def _mutate(cfg, retry_splits: int):
        cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())})
        cfg.setdefault('solver', {}).update(
            {
                'dt': 0.2,
                't_end': 0.2,
                'save_every': 1,
                'min_tau_p_s': 1.0,
                'max_wall_hits_per_step': 1,
                'min_remaining_dt_ratio': 0.0,
                'max_hits_retry_splits': int(retry_splits),
            }
        )
        cfg.setdefault('process', {}).setdefault('step_defaults', {}).setdefault('wall', {}).update(
            {'mode': 'specular', 'stick_probability_scale': 0.0, 'restitution': 1.0, 'diffuse_fraction': 0.0}
        )
        cfg.setdefault('output', {}).update({'write_collision_diagnostics': 1})

    cfg_off_dir = tmp_path / 'cfg_retry_off_3d'
    cfg_on_dir = tmp_path / 'cfg_retry_on_3d'
    cfg_off_dir.mkdir(parents=True, exist_ok=True)
    cfg_on_dir.mkdir(parents=True, exist_ok=True)
    cfg_off = _write_config(cfg_off_dir, ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml', mutate=lambda cfg: _mutate(cfg, 0))
    cfg_on = _write_config(cfg_on_dir, ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml', mutate=lambda cfg: _mutate(cfg, 2))

    out_off = tmp_path / 'out_retry_off_3d'
    out_on = tmp_path / 'out_retry_on_3d'
    run_solver_3d_from_yaml(cfg_off, output_dir=out_off)
    run_solver_3d_from_yaml(cfg_on, output_dir=out_on)

    diag_off = json.loads((out_off / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    diag_on = json.loads((out_on / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    assert int(diag_off['max_hits_retry_splits']) == 0
    assert int(diag_on['max_hits_retry_splits']) == 2
    assert int(diag_on['max_hits_retry_count']) <= 2
    assert int(diag_on['max_hits_reached_count']) <= int(diag_off['max_hits_reached_count'])
    assert int(diag_off['max_hits_retry_exhausted_count']) == 0
    assert float(diag_off['dropped_remaining_dt_total_s']) == pytest.approx(0.0, abs=1e-15)
    assert int(diag_on['max_hits_retry_exhausted_count']) <= int(diag_on['max_hits_reached_count'])
    assert float(diag_on['dropped_remaining_dt_total_s']) >= 0.0


def test_max_hits_retry_local_adaptive_improves_or_matches_3d_max_hit_drop_metrics(tmp_path: Path):
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
    particles_path = tmp_path / 'retry_local_adaptive_particles_3d.csv'
    particles.to_csv(particles_path, index=False)

    def _mutate(cfg, local_adaptive_enabled: int):
        cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())})
        cfg.setdefault('solver', {}).update(
            {
                'dt': 0.2,
                't_end': 0.2,
                'save_every': 1,
                'min_tau_p_s': 1.0,
                'max_wall_hits_per_step': 1,
                'min_remaining_dt_ratio': 0.0,
                'max_hits_retry_splits': 2,
                'adaptive_substep_enabled': 0,
                'max_hits_retry_local_adaptive_enabled': int(local_adaptive_enabled),
                'adaptive_substep_tau_ratio': 0.01,
                'adaptive_substep_max_splits': 3,
            }
        )
        cfg.setdefault('process', {}).setdefault('step_defaults', {}).setdefault('wall', {}).update(
            {'mode': 'specular', 'stick_probability_scale': 0.0, 'restitution': 1.0, 'diffuse_fraction': 0.0}
        )
        cfg.setdefault('output', {}).update({'write_collision_diagnostics': 1})

    cfg_off_dir = tmp_path / 'cfg_retry_local_adaptive_off_3d'
    cfg_on_dir = tmp_path / 'cfg_retry_local_adaptive_on_3d'
    cfg_off_dir.mkdir(parents=True, exist_ok=True)
    cfg_on_dir.mkdir(parents=True, exist_ok=True)
    cfg_off = _write_config(cfg_off_dir, ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml', mutate=lambda cfg: _mutate(cfg, 0))
    cfg_on = _write_config(cfg_on_dir, ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml', mutate=lambda cfg: _mutate(cfg, 1))

    out_off = tmp_path / 'out_retry_local_adaptive_off_3d'
    out_on = tmp_path / 'out_retry_local_adaptive_on_3d'
    run_solver_3d_from_yaml(cfg_off, output_dir=out_off)
    run_solver_3d_from_yaml(cfg_on, output_dir=out_on)

    diag_off = json.loads((out_off / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    diag_on = json.loads((out_on / 'collision_diagnostics.json').read_text(encoding='utf-8'))
    assert int(diag_off['adaptive_substep_enabled']) == 0
    assert int(diag_on['adaptive_substep_enabled']) == 0
    assert int(diag_off['max_hits_retry_local_adaptive_enabled']) == 0
    assert int(diag_on['max_hits_retry_local_adaptive_enabled']) == 1
    assert int(diag_on['max_hits_reached_count']) <= int(diag_off['max_hits_reached_count'])
    assert int(diag_on['unresolved_crossing_count']) == 0
    assert int(diag_on['max_hits_retry_count']) >= 1
    assert int(diag_off['adaptive_substep_segments_count']) == 0
    assert int(diag_off['adaptive_substep_trigger_count']) == 0
    assert int(diag_on['adaptive_substep_segments_count']) > 0
    assert int(diag_on['adaptive_substep_trigger_count']) > 0


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
    assert int(diag['max_hits_retry_splits']) == 0
    assert int(diag['max_hits_retry_local_adaptive_enabled']) == 0
    assert int(diag['adaptive_substep_segments_count']) == 0
    assert int(diag['adaptive_substep_trigger_count']) == 0
    assert int(diag['max_hits_retry_count']) == 0
    assert int(diag['max_hits_retry_exhausted_count']) == 0
    assert float(diag['dropped_remaining_dt_total_s']) == pytest.approx(0.0, abs=1e-15)


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
