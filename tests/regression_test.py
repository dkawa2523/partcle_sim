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
from particle_tracer_unified.solvers.forces import ForceRuntimeParameters
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
    RuntimeState,
    SolverRuntimeOptions,
    run_prepared_runtime,
)
from particle_tracer_unified.solvers.diagnostics import initial_collision_diagnostics
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


def _solver_options_for_test(**overrides) -> SolverRuntimeOptions:
    values = {
        'dt': 0.01,
        't_end': 0.01,
        'base_save_every': 1,
        'plot_limit': 0,
        'rng_seed': 1,
        'max_wall_hits_per_step': 2,
        'min_remaining_dt_ratio': 0.0,
        'adaptive_substep_enabled': 0,
        'adaptive_substep_tau_ratio': 1.0,
        'adaptive_substep_max_splits': 0,
        'epsilon_offset_m': 1.0e-6,
        'on_boundary_tol_m': 1.0e-6,
        'write_collision_diagnostics': 1,
        'valid_mask_policy': 'retry_then_stop',
        'output_options': RuntimeOutputOptions(),
        'drag_model_mode': DRAG_MODEL_STOKES,
        'drag_model_name': 'stokes',
        'contact_tangent_motion_enabled': True,
    }
    values.update(overrides)
    return SolverRuntimeOptions(**values)


def _runtime_state_for_test(
    *,
    x: np.ndarray,
    v: np.ndarray,
    released: bool = True,
    active: bool = True,
    contact_sliding: bool = False,
    contact_endpoint_stopped: bool = False,
    contact_edge_index: int = -1,
    contact_part_id: int = 0,
    contact_normal: np.ndarray | None = None,
    **overrides,
) -> RuntimeState:
    positions = np.asarray(x, dtype=np.float64)
    velocities = np.asarray(v, dtype=np.float64)
    count, spatial_dim = positions.shape
    normal = (
        np.zeros((count, spatial_dim), dtype=np.float64)
        if contact_normal is None
        else np.asarray(contact_normal, dtype=np.float64)
    )
    values = {
        'x': positions.copy(),
        'v': velocities.copy(),
        'released': np.full(count, bool(released), dtype=bool),
        'active': np.full(count, bool(active), dtype=bool),
        'stuck': np.zeros(count, dtype=bool),
        'absorbed': np.zeros(count, dtype=bool),
        'contact_sliding': np.full(count, bool(contact_sliding), dtype=bool),
        'contact_endpoint_stopped': np.full(count, bool(contact_endpoint_stopped), dtype=bool),
        'contact_edge_index': np.full(count, int(contact_edge_index), dtype=np.int32),
        'contact_part_id': np.full(count, int(contact_part_id), dtype=np.int32),
        'contact_normal': normal,
        'escaped': np.zeros(count, dtype=bool),
        'invalid_mask_stopped': np.zeros(count, dtype=bool),
        'numerical_boundary_stopped': np.zeros(count, dtype=bool),
        'invalid_stop_reason_code': np.zeros(count, dtype=np.uint8),
        'save_positions': [],
        'save_meta': [],
        'wall_rows': [],
        'coating_summary_rows': [],
        'max_hit_rows': [],
        'step_rows': [],
        'wall_law_counts': {},
        'wall_summary_counts': {},
        'collision_diagnostics': initial_collision_diagnostics(),
        'rng': np.random.default_rng(1),
        'prev_step_name': None,
        'step_local_counter': 0,
        'save_index': 0,
        'x_trial': np.zeros((count, spatial_dim), dtype=np.float64),
        'v_trial': np.zeros((count, spatial_dim), dtype=np.float64),
        'x_mid_trial': np.zeros((count, spatial_dim), dtype=np.float64),
        'substep_counts': np.ones(count, dtype=np.int32),
        'valid_mask_status_flags': np.zeros(count, dtype=np.uint8),
        'valid_mask_mixed_seen': np.zeros(count, dtype=bool),
        'valid_mask_hard_seen': np.zeros(count, dtype=bool),
    }
    values.update(overrides)
    return RuntimeState(**values)


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


def _write_minimal_config(
    config_dir: Path,
    *,
    spatial_dim: int = 2,
    path_updates: Mapping[str, Any] | None = None,
    solver_updates: Mapping[str, Any] | None = None,
    output_updates: Mapping[str, Any] | None = None,
    geometry_updates: Mapping[str, Any] | None = None,
    provider_updates: Mapping[str, Mapping[str, Any]] | None = None,
    input_contract_updates: Mapping[str, Any] | None = None,
    provider_contract_updates: Mapping[str, Any] | None = None,
) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)

    def mutate(cfg):
        if path_updates:
            cfg.setdefault('paths', {}).update(dict(path_updates))
        if solver_updates:
            cfg.setdefault('solver', {}).update(dict(solver_updates))
        if output_updates:
            cfg.setdefault('output', {}).update(dict(output_updates))
        if geometry_updates:
            cfg.setdefault('providers', {}).setdefault('geometry', {}).update(dict(geometry_updates))
        if provider_updates:
            providers = cfg.setdefault('providers', {})
            for name, provider_cfg in provider_updates.items():
                providers[name] = dict(provider_cfg)
        if input_contract_updates:
            cfg.setdefault('input_contract', {}).update(dict(input_contract_updates))
        if provider_contract_updates:
            cfg.setdefault('provider_contract', {}).update(dict(provider_contract_updates))

    template = ROOT / 'examples' / f'minimal_{int(spatial_dim)}d' / 'run_config.yaml'
    return _write_config(config_dir, template, mutate=mutate)


def _write_minimal_2d_config(
    config_dir: Path,
    *,
    path_updates: Mapping[str, Any] | None = None,
    solver_updates: Mapping[str, Any] | None = None,
    output_updates: Mapping[str, Any] | None = None,
    geometry_updates: Mapping[str, Any] | None = None,
    provider_updates: Mapping[str, Mapping[str, Any]] | None = None,
    input_contract_updates: Mapping[str, Any] | None = None,
    provider_contract_updates: Mapping[str, Any] | None = None,
) -> Path:
    return _write_minimal_config(
        config_dir,
        spatial_dim=2,
        path_updates=path_updates,
        solver_updates=solver_updates,
        output_updates=output_updates,
        geometry_updates=geometry_updates,
        provider_updates=provider_updates,
        input_contract_updates=input_contract_updates,
        provider_contract_updates=provider_contract_updates,
    )


def _write_rows_csv(path: Path, rows: list[Mapping[str, object]]) -> Path:
    pd.DataFrame([dict(row) for row in rows]).to_csv(path, index=False)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _solver_report(out_dir: Path) -> dict[str, Any]:
    return _read_json(out_dir / 'solver_report.json')


def _collision_diagnostics(out_dir: Path) -> dict[str, Any]:
    return _read_json(out_dir / 'collision_diagnostics.json')


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _final_particles(out_dir: Path, *, sort: bool = False) -> pd.DataFrame:
    frame = _read_table(out_dir / 'final_particles.csv')
    if sort:
        return frame.sort_values('particle_id').reset_index(drop=True)
    return frame


def _one_particle_row(
    *,
    spatial_dim: int,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    vx: float = 1.0,
    vy: float = 0.0,
    diameter: float = 1.0e-4,
    density: float = 1000.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        'particle_id': 1,
        'x': float(x),
        'y': float(y),
        'vx': float(vx),
        'vy': float(vy),
        'release_time': 0.0,
        'mass': 1e-15,
        'diameter': float(diameter),
        'density': float(density),
        'charge': 0.0,
        'source_part_id': 10,
        'material_id': 1,
        'source_event_tag': '',
        'stick_probability': 0.0,
    }
    if int(spatial_dim) == 3:
        row.update({'z': float(z), 'vz': 0.0})
    return row


def _write_particle_row(path: Path, row: Mapping[str, object]) -> Path:
    return _write_rows_csv(path, [row])


def _run_solver_for_dim(spatial_dim: int, config_path: Path, output_dir: Path):
    runner = run_solver_3d_from_yaml if int(spatial_dim) == 3 else run_solver_2d_from_yaml
    return runner(config_path, output_dir=output_dir)


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


def _regular_axes(spatial_dim: int = 2) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray([0.0, 0.5, 1.0], dtype=np.float64) for _ in range(int(spatial_dim)))


def _regular_valid_mask(spatial_dim: int = 2, *, fill: bool = True) -> np.ndarray:
    return np.full(tuple(3 for _ in range(int(spatial_dim))), bool(fill), dtype=bool)


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


def _field_provider_with_mismatched_velocity_time_axes() -> FieldProviderND:
    axes = _regular_axes(2)
    return FieldProviderND(
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
            valid_mask=_regular_valid_mask(2),
            time_mode='transient',
            metadata={'provider_kind': 'precomputed_npz'},
        ),
        kind='precomputed_npz',
    )


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


def _state_frame(rows: list[tuple[int, str]]) -> pd.DataFrame:
    state_columns = ('active', 'stuck', 'absorbed', 'escaped', 'invalid_mask_stopped')
    out = []
    for particle_id, state in rows:
        row = {'particle_id': int(particle_id), **{name: 0 for name in state_columns}}
        if state == 'active_free_flight':
            row['active'] = 1
        elif state in state_columns:
            row[state] = 1
        else:
            raise ValueError(f'unknown test particle state: {state}')
        out.append(row)
    return pd.DataFrame(out)


def _square_boundary_edges() -> np.ndarray:
    return np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )


def _square_boundary_runtime(part_ids: tuple[int, int, int, int] = (1, 2, 3, 4)) -> SimpleNamespace:
    edges = _square_boundary_edges()
    geom = SimpleNamespace(
        spatial_dim=2,
        boundary_edges=edges,
        boundary_edge_part_ids=np.asarray(part_ids, dtype=np.int32),
    )
    return SimpleNamespace(geometry_provider=SimpleNamespace(geometry=geom))


def _square_boundary_service_runtime(
    part_ids: tuple[int, int, int, int] = (10, 20, 30, 40),
) -> SimpleNamespace:
    edges = _square_boundary_edges()
    grid_shape = (11, 11)
    geometry = SimpleNamespace(
        spatial_dim=2,
        axes=(np.linspace(0.0, 1.0, grid_shape[0]), np.linspace(0.0, 1.0, grid_shape[1])),
        boundary_edges=edges,
        boundary_edge_part_ids=np.asarray(part_ids, dtype=np.int32),
        boundary_loops_2d=build_boundary_loops_2d(edges),
        sdf=np.zeros(grid_shape, dtype=np.float64),
        nearest_boundary_part_id_map=np.zeros(grid_shape, dtype=np.int32),
        normal_components=(
            np.zeros(grid_shape, dtype=np.float64),
            np.ones(grid_shape, dtype=np.float64),
        ),
    )
    return SimpleNamespace(geometry_provider=SimpleNamespace(geometry=geometry), field_provider=None)


def _box_field_runtime(spatial_dim: int = 2) -> SimpleNamespace:
    axis = np.asarray([0.0, 1.0], dtype=np.float64)
    return SimpleNamespace(
        wall_catalog=None,
        geometry_provider=None,
        field_provider=SimpleNamespace(
            field=SimpleNamespace(axes=tuple(axis.copy() for _ in range(int(spatial_dim))))
        ),
    )


def _wall_hit_context(*, particle_id: int = 42, spatial_dim: int = 2) -> dict[str, object]:
    return {
        'runtime': _box_field_runtime(spatial_dim),
        'step': ProcessStepRow(step_id=1, step_name='run', start_s=0.0, end_s=1.0),
        'particles': SimpleNamespace(
            particle_id=np.asarray([int(particle_id)], dtype=np.int64),
            stick_probability=np.asarray([0.0], dtype=np.float64),
        ),
        'collision_diagnostics': initial_collision_diagnostics(),
        'max_hit_rows': [],
        'wall_rows': [],
        'coating_summary_rows': [],
        'wall_law_counts': {},
        'wall_summary_counts': {},
        'stuck': np.asarray([False], dtype=bool),
        'absorbed': np.asarray([False], dtype=bool),
        'active': np.asarray([True], dtype=bool),
    }


def _apply_test_wall_hit_step(
    context: Mapping[str, object] | None = None,
    **overrides,
) -> WallHitStepResult:
    hit = np.asarray(overrides.get('hit', [0.0, 0.5]), dtype=np.float64)
    spatial_dim = int(hit.size)
    values: dict[str, object] = (
        dict(context) if context is not None else _wall_hit_context(spatial_dim=spatial_dim)
    )
    values.update(
        {
            'particle_index': 0,
            'rng': np.random.default_rng(123),
            'hit': hit,
            'n_out': np.asarray([-1.0] + [0.0] * (spatial_dim - 1), dtype=np.float64),
            'hit_dt': 0.0,
            'part_id': 7,
            'v_hit': np.asarray([-2.0] + [0.0] * (spatial_dim - 1), dtype=np.float64),
            'remaining_dt': 0.2,
            'segment_dt': 0.2,
            'hit_count': 0,
            'total_hit_count': 0,
            'hit_part_ids': [],
            'hit_outcomes': [],
            'max_wall_hits_per_step': 1,
            'min_remaining_dt': 0.0,
            'epsilon_offset_m': 1.0e-6,
            'on_boundary_tol_m': 1.0e-6,
            't': 0.2,
            'triangle_surface_3d': None,
        }
    )
    values.update(overrides)
    for name in ('hit', 'n_out', 'v_hit'):
        values[name] = np.asarray(values[name], dtype=np.float64)
    return _apply_wall_hit_step(**values)


def _regular_grid_adaptive_substep_count_for_drag_model(drag_model_mode: int) -> int:
    axes = (
        np.asarray([0.0, 500.0, 1000.0], dtype=np.float64),
        np.asarray([0.0, 500.0, 1000.0], dtype=np.float64),
    )
    valid_mask = np.ones((3, 3), dtype=bool)
    quantities = {
        'ux': np.zeros((3, 3), dtype=np.float64),
        'uy': np.zeros((3, 3), dtype=np.float64),
        'rho_g': np.ones((3, 3), dtype=np.float64) * 10.0,
        'mu': np.ones((3, 3), dtype=np.float64) * 1.0e-3,
        'T': np.ones((3, 3), dtype=np.float64) * 300.0,
    }
    field_provider = _regular_field_provider_from_arrays(axes, valid_mask, quantities=quantities)
    geometry_provider = _geometry_provider_from_arrays(
        axes,
        valid_mask,
        sdf=-np.ones_like(valid_mask, dtype=np.float64),
        normal_components=(np.zeros_like(valid_mask, dtype=np.float64), np.ones_like(valid_mask, dtype=np.float64)),
    )
    runtime = SimpleNamespace(
        geometry_provider=geometry_provider,
        field_provider=field_provider,
        gas=SimpleNamespace(density_kgm3=10.0, dynamic_viscosity_Pas=1.0e-3, temperature=300.0),
        config_payload={'solver': {'field_backend_mode': 'regular_grid'}},
    )
    compiled = _compile_runtime_arrays(runtime, spatial_dim=2)
    x = np.asarray([[500.0, 500.0]], dtype=np.float64)
    v = np.asarray([[100.0, 0.0]], dtype=np.float64)
    density = np.asarray([1000.0], dtype=np.float64)
    diameter = np.asarray([1.0], dtype=np.float64)
    mass = density * np.pi * diameter**3 / 6.0
    substeps = np.zeros(1, dtype=np.int32)

    _advance_trial_particles(
        spatial_dim=2,
        compiled=compiled,
        x=x,
        v=v,
        active=np.asarray([True], dtype=bool),
        tau_p=np.asarray([1.0], dtype=np.float64),
        particle_diameter=diameter,
        particle_mass=mass,
        particle_density=density,
        dep_particle_rel_permittivity=np.asarray([np.nan], dtype=np.float64),
        thermophoretic_coeff=np.asarray([np.nan], dtype=np.float64),
        flow_scale_particle=np.asarray([1.0], dtype=np.float64),
        drag_scale_particle=np.asarray([1.0], dtype=np.float64),
        body_scale_particle=np.asarray([1.0], dtype=np.float64),
        t=1.0,
        dt_step=1.0,
        phys={
            'flow_scale': 1.0,
            'drag_tau_scale': 1.0,
            'body_accel_scale': 1.0,
            'min_tau_p_s': 1.0e-12,
            'gas_density_kgm3': 10.0,
            'gas_mu_pas': 1.0e-3,
            'gas_temperature_K': 300.0,
            'gas_molecular_mass_kg': 60.0 * 1.66053906660e-27,
        },
        body_accel=np.zeros(2, dtype=np.float64),
        gas_density_kgm3=10.0,
        gas_mu_pas=1.0e-3,
        drag_model_mode=int(drag_model_mode),
        integrator_mode=int(get_integrator_spec('etd').mode),
        adaptive_substep_enabled=1,
        adaptive_substep_tau_ratio=0.25,
        adaptive_substep_max_splits=4,
        x_trial=np.zeros_like(x),
        v_trial=np.zeros_like(v),
        x_mid_trial=np.zeros_like(x),
        substep_counts=substeps,
        valid_mask_status_flags=np.zeros(1, dtype=np.uint8),
    )
    return int(substeps[0])


def _write_invalid_left_field_bundle(path: Path, *, invalid_until_x: float) -> Path:
    axes = np.linspace(-1.0, 1.0, 81)
    _write_field_bundle(path, axes, axes)
    payload = {key: value for key, value in np.load(path).items()}
    valid_mask = np.ones((axes.size, axes.size), dtype=bool)
    valid_mask[axes <= float(invalid_until_x), :] = False
    payload['valid_mask'] = valid_mask
    np.savez_compressed(path, **payload)
    return path


def _write_transient_invalid_left_field_bundle(path: Path, *, invalid_until_x: float) -> Path:
    axes = np.linspace(-1.0, 1.0, 81)
    shape = (axes.size, axes.size)
    xx, yy = np.meshgrid(axes, axes, indexing='ij')
    times = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    valid_mask = np.ones(shape, dtype=bool)
    valid_mask[axes <= float(invalid_until_x), :] = False
    np.savez_compressed(
        path,
        axis_0=axes,
        axis_1=axes,
        times=times,
        valid_mask=valid_mask,
        ux=np.stack([(1.0 + t) * np.ones(shape, dtype=np.float64) for t in times], axis=0),
        uy=np.stack([0.05 * np.cos(xx * 10.0) + 0.0 * yy for _ in times], axis=0),
        mu=np.stack([1.8e-5 * np.ones(shape, dtype=np.float64) for _ in times], axis=0),
    )
    return path


def _write_contract_particle(path: Path, *, release_time: float) -> Path:
    pd.DataFrame(
        [
            {
                'particle_id': 1,
                'x': -0.8,
                'y': -0.2,
                'vx': 0.0,
                'vy': 0.0,
                'release_time': float(release_time),
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
    ).to_csv(path, index=False)
    return path


def _write_precomputed_field_config(
    cfg_dir: Path,
    *,
    field_path: Path,
    particles_path: Path | None = None,
    solver_updates: Mapping[str, object] | None = None,
    output_updates: Mapping[str, object] | None = None,
    input_mode: str | None = None,
    provider_contract: Mapping[str, object] | None = None,
    source_preprocess_enabled: bool | None = None,
) -> Path:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    def mutate(cfg: dict[str, Any]) -> None:
        if particles_path is not None:
            cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())})
        cfg.setdefault('providers', {}).update(
            {'field': {'kind': 'precomputed_npz', 'npz_path': str(field_path.resolve())}}
        )
        if solver_updates is not None:
            cfg.setdefault('solver', {}).update(dict(solver_updates))
        if output_updates is not None:
            cfg.setdefault('output', {}).update(dict(output_updates))
        if input_mode is not None:
            cfg.setdefault('input_contract', {}).update({'initial_particle_field_support': str(input_mode)})
        if provider_contract is not None:
            cfg.setdefault('provider_contract', {}).update(dict(provider_contract))
        if source_preprocess_enabled is not None:
            cfg.setdefault('source', {}).setdefault('preprocess', {}).update(
                {'enabled': bool(source_preprocess_enabled)}
            )

    return _write_config(
        cfg_dir,
        ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml',
        mutate=mutate,
    )


def _final_xy_velocity(out_dir: Path) -> np.ndarray:
    df = _final_particles(out_dir)
    return df[['x', 'y', 'v_x', 'v_y']].to_numpy(dtype=np.float64)


def _write_stochastic_motion_config(
    config_dir: Path,
    *,
    solver_seed: int = 1234,
    stochastic_motion: Mapping[str, Any] | None = None,
) -> Path:
    solver: dict[str, Any] = {'t_end': 0.06, 'save_every': 1, 'seed': int(solver_seed)}
    if stochastic_motion is not None:
        solver['stochastic_motion'] = dict(stochastic_motion)
    return _write_minimal_2d_config(
        config_dir,
        solver_updates=solver,
        output_updates={'artifact_mode': 'minimal'},
    )


def _write_boundary_diagnostics_case(tmp_path: Path) -> tuple[Path, Path]:
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
    boundary_edges = _square_boundary_edges()
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
    return case_dir, output_dir


def _run_minimal_case(spatial_dim: int, output_dir: Path):
    config_path = ROOT / 'examples' / f'minimal_{spatial_dim}d' / 'run_config.yaml'
    return _run_solver_for_dim(spatial_dim, config_path, output_dir)


def _write_particle_solver_config(
    config_dir: Path,
    *,
    spatial_dim: int,
    particles_path: Path,
    solver_updates: Mapping[str, object] | None = None,
    geometry_updates: Mapping[str, object] | None = None,
    field_updates: Mapping[str, object] | None = None,
    output_updates: Mapping[str, object] | None = None,
) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)

    def mutate(cfg: dict[str, Any]) -> None:
        cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())})
        if geometry_updates:
            cfg.setdefault('providers', {}).setdefault('geometry', {}).update(dict(geometry_updates))
        if field_updates:
            cfg.setdefault('providers', {}).setdefault('field', {}).update(dict(field_updates))
        if solver_updates:
            cfg.setdefault('solver', {}).update(dict(solver_updates))
        if output_updates:
            cfg.setdefault('output', {}).update(dict(output_updates))

    return _write_config(
        config_dir,
        ROOT / 'examples' / f'minimal_{spatial_dim}d' / 'run_config.yaml',
        mutate=mutate,
    )


def _final_particle_position(out_dir: Path, spatial_dim: int) -> np.ndarray:
    columns = ['x', 'y', 'z'][: int(spatial_dim)]
    return _final_particles(out_dir).loc[0, columns].to_numpy(dtype=float)


def _write_integrator_config(
    config_dir: Path,
    *,
    spatial_dim: int,
    particles_path: Path,
    integrator: str,
    dt: float,
    bounds: list[float],
    grid_shape: list[int],
    shear_rate: float,
) -> Path:
    return _write_particle_solver_config(
        config_dir,
        spatial_dim=spatial_dim,
        particles_path=particles_path,
        geometry_updates={'bounds': list(bounds), 'grid_shape': list(grid_shape)},
        field_updates={'shear_rate': float(shear_rate)},
        solver_updates={
            'integrator': str(integrator),
            'dt': float(dt),
            't_end': 0.2,
            'save_every': 1,
            'min_tau_p_s': 1.0e-8,
        },
    )

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


def _write_single_reflection_case(tmp_path: Path, *, spatial_dim: int) -> Path:
    particle_row = {
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
    if int(spatial_dim) == 3:
        particle_row.update({'z': 0.0, 'vz': 0.0})
    particles_path = tmp_path / f'single_reflection_{spatial_dim}d.csv'
    pd.DataFrame([particle_row]).to_csv(particles_path, index=False)
    part_walls_path = tmp_path / f'part_walls_reflect_{spatial_dim}d.csv'
    pd.DataFrame(
        [
            {
                'part_id': 10,
                'part_name': 'wall_10',
                'material_id': 1,
                'material_name': 'steel',
                'wall_law': 'specular',
                'wall_restitution': 1.0,
                'wall_diffuse_fraction': 0.0,
                'wall_stick_probability': 0.0,
            },
            {
                'part_id': 20,
                'part_name': 'wall_20',
                'material_id': 2,
                'material_name': 'ceramic',
                'wall_law': 'specular',
                'wall_restitution': 1.0,
                'wall_diffuse_fraction': 0.0,
                'wall_stick_probability': 0.0,
            },
        ]
    ).to_csv(part_walls_path, index=False)

    def mutate(cfg: dict[str, Any]) -> None:
        cfg.setdefault('paths', {}).update(
            {
                'particles_csv': str(particles_path.resolve()),
                'part_walls_csv': str(part_walls_path.resolve()),
            }
        )
        cfg.setdefault('providers', {}).setdefault('field', {}).update({'shear_rate': 0.0})
        cfg.setdefault('solver', {}).update(
            {
                'integrator': 'etd',
                'dt': 0.2,
                't_end': 0.2,
                'save_every': 1,
                'min_tau_p_s': 1.0e-8,
            }
        )

    return _write_config(
        tmp_path,
        ROOT / 'examples' / f'minimal_{spatial_dim}d' / 'run_config.yaml',
        mutate=mutate,
    )


def _assert_single_reflection_result(out_dir: Path) -> None:
    final_df = _final_particles(out_dir)
    x_final = float(final_df.loc[0, 'x'])
    vx_final = float(final_df.loc[0, 'v_x'])
    expected_x, expected_vx = _single_reflection_expected_zero_flow(
        x0=0.9,
        wall_x=1.0,
        v0=2.0,
        tau=0.1,
        dt=0.2,
    )
    assert x_final == pytest.approx(expected_x, abs=2.0e-4)
    assert vx_final == pytest.approx(expected_vx, abs=2.0e-4)


def _write_wall_bounce_config(
    tmp_path: Path,
    *,
    spatial_dim: int,
    name: str,
    vx: float,
    max_wall_hits_per_step: int,
) -> Path:
    particles_path = _write_particle_row(
        tmp_path / f'{name}_particles_{spatial_dim}d.csv',
        _one_particle_row(spatial_dim=spatial_dim, vx=vx, diameter=1.0e-6, density=1200.0),
    )
    return _write_particle_solver_config(
        tmp_path / f'{name}_cfg_{spatial_dim}d',
        spatial_dim=spatial_dim,
        particles_path=particles_path,
        solver_updates={
            'dt': 0.2,
            't_end': 0.2,
            'save_every': 1,
            'min_tau_p_s': 1.0,
            'max_wall_hits_per_step': int(max_wall_hits_per_step),
            'min_remaining_dt_ratio': 0.0,
        },
        output_updates={'write_collision_diagnostics': 1},
    )
