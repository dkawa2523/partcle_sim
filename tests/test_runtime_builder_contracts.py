from __future__ import annotations

import regression_helpers as _regression_helpers
from particle_tracer_unified.core.coordinate_systems import ring_area_weight
from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    GasProperties,
    ParticleTable,
    PreparedRuntime,
    QuantitySeriesND,
    RegularFieldND,
    RuntimeLike,
    SourcePreprocessResult,
)
from particle_tracer_unified.core.input_contract import enforce_initial_particle_field_support
from particle_tracer_unified.io.runtime_builder import prepare_runtime, prepared_runtime_summary
from particle_tracer_unified.solvers.runtime_plan import build_solver_plan
from particle_tracer_unified.solvers.source_preprocess import boundary_service_for_source_preprocess

globals().update({
    name: value
    for name, value in vars(_regression_helpers).items()
    if not name.startswith("__")
})

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

def test_axisymmetric_rz_runtime_reports_coordinate_and_axis_names():
    config_dir = ROOT / 'examples' / 'minimal_2d'
    cfg = yaml.safe_load((config_dir / 'run_config.yaml').read_text(encoding='utf-8'))
    cfg['run']['coordinate_system'] = 'axisymmetric_rz'
    cfg['providers']['geometry']['bounds'] = [0.0, 1.0, -1.0, 1.0]
    _absolutize_paths(cfg, config_dir)

    runtime = build_runtime_from_config(cfg, config_dir)
    prepared = prepare_runtime(runtime, seed=1)
    summary = prepared_runtime_summary(prepared)
    input_report = build_initial_particle_field_support_report(prepared)
    provider_report = build_boundary_field_support_report(prepared)

    assert runtime.coordinate_system == 'axisymmetric_rz'
    assert runtime.config_payload['run']['coordinate_system'] == 'axisymmetric_rz'
    assert summary['coordinate_system'] == 'axisymmetric_rz'
    assert summary['axis_names'] == ['r', 'z']
    assert summary['axisymmetric_rz']['r0_axis_boundary_edge_count'] == 1
    assert summary['axisymmetric_rz']['semantics'] == '2d_meridional_rz'
    assert summary['axisymmetric_rz']['velocity_components'] == ['v_r', 'v_z']
    assert summary['axisymmetric_rz']['v_theta_dynamics'] == 'out_of_scope'
    assert summary['axisymmetric_rz']['source_ring_weighting_policy'] == 'not_applied_implicitly'
    assert summary['field_provider']['axis_names'] == ['r', 'z']
    assert summary['geometry_provider']['axis_names'] == ['r', 'z']
    assert summary['geometry_provider']['axisymmetric_rz']['r0_on_grid'] == 1
    assert summary['geometry_provider']['axisymmetric_rz']['r0_axis_boundary_edge_count'] == 1
    assert summary['geometry_provider']['axisymmetric_rz']['axis_boundary_policy'] == 'report_only_collision_unchanged'
    assert input_report['coordinate_system'] == 'axisymmetric_rz'
    assert input_report['axis_names'] == ['r', 'z']
    assert input_report['axisymmetric_rz']['r0_axis_boundary_edge_count'] == 1
    assert provider_report['coordinate_system'] == 'axisymmetric_rz'
    assert provider_report['axis_names'] == ['r', 'z']
    assert provider_report['axisymmetric_rz']['r0_axis_boundary_edge_count'] == 1
    assert provider_report['geometry_boundary']['axisymmetric_rz']['r0_axis_boundary_edge_count'] == 1
    assert provider_report['field_support']['axis_names'] == ['r', 'z']

def test_axisymmetric_ring_area_weight_known_inputs():
    weights = ring_area_weight(np.asarray([0.0, 0.5, 1.0], dtype=np.float64))
    np.testing.assert_allclose(weights, np.asarray([0.0, np.pi, 2.0 * np.pi], dtype=np.float64))
    assert ring_area_weight(0.25) == pytest.approx(0.5 * np.pi)

    with pytest.raises(ValueError, match='non-negative'):
        ring_area_weight(np.asarray([-1.0], dtype=np.float64))
    with pytest.raises(ValueError, match='finite'):
        ring_area_weight(np.asarray([np.nan], dtype=np.float64))

def test_axisymmetric_rz_runtime_rejects_negative_radial_axis():
    config_dir = ROOT / 'examples' / 'minimal_2d'
    cfg = yaml.safe_load((config_dir / 'run_config.yaml').read_text(encoding='utf-8'))
    cfg['run']['coordinate_system'] = 'axisymmetric_rz'
    cfg['providers']['geometry']['bounds'] = [-1.0, 1.0, -1.0, 1.0]
    _absolutize_paths(cfg, config_dir)

    with pytest.raises(ValueError, match='axisymmetric_rz radial axis_0 must be non-negative'):
        build_runtime_from_config(cfg, config_dir)

def test_cartesian_xy_runtime_axis_names_remain_unchanged():
    config_dir = ROOT / 'examples' / 'minimal_2d'
    cfg = yaml.safe_load((config_dir / 'run_config.yaml').read_text(encoding='utf-8'))
    _absolutize_paths(cfg, config_dir)

    runtime = build_runtime_from_config(cfg, config_dir)
    prepared = prepare_runtime(runtime, seed=1)
    summary = prepared_runtime_summary(prepared)

    assert runtime.coordinate_system == 'cartesian_xy'
    assert summary['axis_names'] == ['x', 'y']
    assert summary['field_provider']['axis_names'] == ['x', 'y']
    assert 'axisymmetric_rz' not in summary

def _axisymmetric_positive_r_config(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    config_dir = ROOT / 'examples' / 'minimal_2d'
    particles_path = _write_particle_row(
        tmp_path / 'axisymmetric_positive_r_particles.csv',
        _one_particle_row(spatial_dim=2, x=0.25, y=0.0, vx=0.0, vy=0.0),
    )
    cfg = yaml.safe_load((config_dir / 'run_config.yaml').read_text(encoding='utf-8'))
    cfg['run']['coordinate_system'] = 'axisymmetric_rz'
    cfg['paths']['particles_csv'] = str(particles_path.resolve())
    cfg['providers']['geometry']['bounds'] = [0.0, 1.0, -1.0, 1.0]
    _absolutize_paths(cfg, config_dir)
    return config_dir, cfg


def test_axisymmetric_source_ring_weighting_is_not_implicit(tmp_path: Path):
    config_dir, cfg = _axisymmetric_positive_r_config(tmp_path)

    prepared = prepare_runtime(build_runtime_from_config(cfg, config_dir), seed=1)

    assert prepared.source_preprocess is not None
    ring_summary = prepared.source_preprocess.source_model_summary['axisymmetric_ring_area_weight']
    assert ring_summary['enabled'] == 0
    assert ring_summary['applied_to_sampling'] == 0
    assert ring_summary['policy'] == 'not_applied'
    assert 'axisymmetric_ring_area_weight' not in prepared.source_preprocess.diagnostics_rows[0]


def test_axisymmetric_source_ring_weight_reporting_requires_explicit_config(tmp_path: Path):
    config_dir, cfg = _axisymmetric_positive_r_config(tmp_path)
    cfg.setdefault('source', {}).setdefault('preprocess', {})['ring_area_weighted_source_reporting'] = True

    prepared = prepare_runtime(build_runtime_from_config(cfg, config_dir), seed=1)

    assert prepared.source_preprocess is not None
    row = prepared.source_preprocess.diagnostics_rows[0]
    ring_summary = prepared.source_preprocess.source_model_summary['axisymmetric_ring_area_weight']
    assert row['axisymmetric_radius_m'] == pytest.approx(0.25)
    assert row['axisymmetric_ring_area_weight'] == pytest.approx(0.5 * np.pi)
    assert ring_summary['enabled'] == 1
    assert ring_summary['applied_to_sampling'] == 0
    assert ring_summary['count'] == 1
    assert ring_summary['sum'] == pytest.approx(0.5 * np.pi)

def test_precomputed_field_axis_mismatch_names_axis(tmp_path: Path):
    axes = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    geom_path = _write_precomputed_geometry_npz(
        tmp_path / 'geom_axis_mismatch.npz',
        axes,
        axes,
        valid_mask=np.ones((3, 3), dtype=bool),
    )
    field_path = tmp_path / 'field_axis_mismatch.npz'
    np.savez_compressed(
        field_path,
        axis_0=axes,
        axis_1=axes + 0.1,
        times=np.asarray([0.0], dtype=np.float64),
        valid_mask=np.ones((3, 3), dtype=bool),
        ux=np.zeros((3, 3), dtype=np.float64),
        uy=np.zeros((3, 3), dtype=np.float64),
    )
    cfg = yaml.safe_load((ROOT / 'examples' / 'minimal_2d' / 'run_config.yaml').read_text(encoding='utf-8'))
    cfg['providers']['geometry'] = {'kind': 'precomputed_npz', 'npz_path': str(geom_path.resolve())}
    cfg['providers']['field'] = {'kind': 'precomputed_npz', 'npz_path': str(field_path.resolve())}
    _absolutize_paths(cfg, ROOT / 'examples' / 'minimal_2d')

    with pytest.raises(ValueError, match='axis_1'):
        build_runtime_from_config(cfg, ROOT / 'examples' / 'minimal_2d')

def test_runtime_builder_rejects_recipe_manifest_path(tmp_path: Path):
    config_path = _write_minimal_2d_config(
        tmp_path,
        path_updates={'recipe_manifest_yaml': str((tmp_path / 'recipe_manifest.yaml').resolve())},
    )
    with pytest.raises(ValueError, match='recipe_manifest_yaml is no longer supported'):
        build_prepared_runtime_from_yaml(config_path)

def test_runtime_builder_allows_process_step_gaps_as_time_label_overlay(tmp_path: Path):
    steps_path = _write_rows_csv(
        tmp_path / 'process_steps_gap_allowed.csv',
        [
            {'step_id': 1, 'step_name': 'etch', 'start_s': 0.0, 'end_s': 0.5},
            {'step_id': 2, 'step_name': 'purge', 'start_s': 0.75, 'end_s': 1.0},
        ],
    )
    config_path = _write_minimal_2d_config(
        tmp_path,
        path_updates={'process_steps_csv': str(steps_path.resolve())},
        output_updates={'write_positions': 0, 'write_segmented_positions': 1},
    )

    prepared = build_prepared_runtime_from_yaml(config_path)

    assert prepared.runtime.process_steps is not None
    assert [row.step_name for row in prepared.runtime.process_steps.rows] == ['etch', 'purge']
    assert prepared.runtime.process_steps.active_at(0.6) is None

def test_process_step_override_columns_are_rejected(tmp_path: Path):
    steps_path = _write_rows_csv(
        tmp_path / 'process_steps_with_override_columns.csv',
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
        ],
    )
    config_path = _write_minimal_2d_config(
        tmp_path,
        path_updates={'process_steps_csv': str(steps_path.resolve())},
        solver_updates={'t_end': 0.02, 'save_every': 1, 'plot_particle_limit': 0},
    )

    with pytest.raises(ValueError, match='process_steps.csv supports only time-label columns'):
        build_prepared_runtime_from_yaml(config_path)

def test_runtime_builder_rejects_zero_duration_process_steps(tmp_path: Path):
    steps_path = _write_rows_csv(
        tmp_path / 'process_steps_zero_duration.csv',
        [
            {'step_id': 1, 'step_name': 'instant_marker', 'start_s': 0.5, 'end_s': 0.5},
        ],
    )
    config_path = _write_minimal_2d_config(
        tmp_path,
        path_updates={'process_steps_csv': str(steps_path.resolve())},
    )
    with pytest.raises(ValueError, match='must have end_s > start_s'):
        build_prepared_runtime_from_yaml(config_path)

@pytest.mark.parametrize(
    ('solver_updates', 'message'),
    [
        ({'dt': float('nan')}, 'solver.dt must be finite and > 0'),
        ({'dt': float('inf')}, 'solver.dt must be finite and > 0'),
        ({'t_end': float('nan')}, 'solver.t_end must be finite and >= 0'),
        ({'t_end': float('inf')}, 'solver.t_end must be finite and >= 0'),
    ],
)
def test_solver_rejects_nonfinite_time_controls(
    tmp_path: Path,
    solver_updates: Mapping[str, float],
    message: str,
):
    config_path = _write_minimal_2d_config(
        tmp_path / f"cfg_{next(iter(solver_updates)).replace('_', '')}",
        solver_updates=solver_updates,
        output_updates={'write_positions': 0},
    )
    with pytest.raises(ValueError, match=message):
        run_solver_2d_from_yaml(config_path, output_dir=tmp_path / 'out_nonfinite_time')

def test_step_loop_rejects_non_advancing_step(monkeypatch: pytest.MonkeyPatch):
    from particle_tracer_unified.solvers import high_fidelity_runtime as hfr

    def _no_progress_step(**kwargs):
        return float(kwargs['t'])

    monkeypatch.setattr(hfr, '_advance_runtime_step', _no_progress_step)

    zero = np.zeros(0, dtype=np.float64)
    state = SimpleNamespace(
        solver_plan=SimpleNamespace(
            t_end=1.0e-3,
            dt=1.0e-3,
        )
    )
    with pytest.raises(RuntimeError, match='did not advance time'):
        hfr._run_runtime_step_loop(
            runtime=None,
            particles=None,
            state=state,
            options=None,
            compiled=None,
            boundary_service=None,
            spatial_dim=2,
            n_particles=0,
            mins=np.zeros(2, dtype=np.float64),
            maxs=np.ones(2, dtype=np.float64),
            tau_p=zero,
            particle_mass=zero,
            particle_diameter=zero,
            particle_density=zero,
            particle_id=np.zeros(0, dtype=np.int64),
            source_part_id=np.zeros(0, dtype=np.int32),
            release_time=zero,
            particle_stick_probability=zero,
            dep_particle_rel_permittivity=zero,
            thermophoretic_coeff=zero,
            flow_scale_particle=zero,
            drag_scale_particle=zero,
            body_scale_particle=zero,
        )

def test_solver_plan_rejects_unknown_valid_mask_policy(tmp_path: Path):
    config_path = _write_minimal_2d_config(
        tmp_path / 'cfg_bad_valid_mask_policy',
        solver_updates={'valid_mask_policy': 'warn_and_continue'},
    )
    prepared = build_prepared_runtime_from_yaml(config_path)

    with pytest.raises(ValueError, match='solver.valid_mask_policy'):
        build_solver_plan(prepared, spatial_dim=2)


def test_solver_plan_release_grace_defaults_off_and_validates_enabled(tmp_path: Path):
    default_cfg = _write_minimal_2d_config(tmp_path / 'cfg_release_grace_default')
    default_plan = build_solver_plan(build_prepared_runtime_from_yaml(default_cfg), spatial_dim=2)

    assert bool(default_plan.release_grace.enabled) is False
    assert default_plan.release_grace.grace_time_s == pytest.approx(0.0)
    assert default_plan.release_grace.clearance_m == pytest.approx(
        max(default_plan.on_boundary_tol_m, default_plan.epsilon_offset_m)
    )

    enabled_cfg = _write_minimal_2d_config(
        tmp_path / 'cfg_release_grace_enabled',
        solver_updates={
            'release_grace': {
                'enabled': True,
                'grace_time_s': 2.0e-6,
                'clearance_m': 3.0e-6,
                'min_outward_normal_speed_mps': 0.1,
            }
        },
    )
    enabled_plan = build_solver_plan(build_prepared_runtime_from_yaml(enabled_cfg), spatial_dim=2)
    assert bool(enabled_plan.release_grace.enabled) is True
    assert enabled_plan.release_grace.grace_time_s == pytest.approx(2.0e-6)
    assert enabled_plan.release_grace.clearance_m == pytest.approx(3.0e-6)
    assert enabled_plan.release_grace.min_outward_normal_speed_mps == pytest.approx(0.1)

    bad_cfg = _write_minimal_2d_config(
        tmp_path / 'cfg_release_grace_bad',
        solver_updates={'release_grace': {'enabled': True, 'grace_time_s': 0.0}},
    )
    with pytest.raises(ValueError, match='solver.release_grace.grace_time_s'):
        build_solver_plan(build_prepared_runtime_from_yaml(bad_cfg), spatial_dim=2)


def test_solver_plan_parses_string_boolean_flags(tmp_path: Path):
    false_cfg = _write_minimal_2d_config(
        tmp_path / 'cfg_false_flags',
        solver_updates={
            'adaptive_substep_enabled': 'false',
            'boundary_broad_phase_enabled': 'false',
            'electric_enabled': 'false',
            'forces': {'electric': {'enabled': False}},
        },
    )
    true_cfg = _write_minimal_2d_config(
        tmp_path / 'cfg_true_flags',
        solver_updates={
            'adaptive_substep_enabled': 'true',
            'boundary_broad_phase_enabled': 'true',
            'electric_enabled': 'true',
            'forces': {'electric': {'enabled': False}},
        },
    )

    false_plan = build_solver_plan(build_prepared_runtime_from_yaml(false_cfg), spatial_dim=2)
    true_plan = build_solver_plan(build_prepared_runtime_from_yaml(true_cfg), spatial_dim=2)

    assert int(false_plan.adaptive_substep_enabled) == 0
    assert bool(false_plan.boundary_broad_phase_enabled) is False
    assert bool(false_plan.field_sample.need_electric) is False
    assert int(true_plan.adaptive_substep_enabled) == 1
    assert bool(true_plan.boundary_broad_phase_enabled) is True
    assert bool(true_plan.field_sample.need_electric) is True

def test_runtime_builder_rejects_unresolved_source_event_bindings(tmp_path: Path):
    events_path = _write_rows_csv(
        tmp_path / 'source_events_bad_binding.csv',
        [
            {
                'event_id': 1,
                'event_name': 'bad_binding',
                'event_kind': 'gaussian_burst',
                'enabled': 1,
                'bind_step_name': 'missing_step',
                'time_anchor': 'step_start',
            }
        ],
    )
    config_path = _write_minimal_2d_config(
        tmp_path,
        path_updates={'source_events_csv': str(events_path.resolve())},
    )
    with pytest.raises(ValueError, match='Unresolved source event bindings'):
        build_prepared_runtime_from_yaml(config_path)

def test_source_events_reject_transition_bindings(tmp_path: Path):
    events_path = _write_rows_csv(
        tmp_path / 'source_events_transition.csv',
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
        ],
    )

    with pytest.raises(ValueError, match='transition bindings are no longer supported'):
        load_source_events_csv(events_path)

def test_shared_source_schema_keeps_material_wall_loading_and_defaults_in_sync(tmp_path: Path):
    materials_path = _write_rows_csv(
        tmp_path / 'materials_schema.csv',
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
        ],
    )

    part_walls_path = _write_rows_csv(
        tmp_path / 'part_walls_schema.csv',
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
        ],
    )

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
    assert {
        'source_resuspension_speed_threshold_mps',
        'source_resuspension_tau_threshold_Pa',
        'source_resuspension_utau_threshold_mps',
        'source_resuspension_shear_length_m',
        'source_dynamic_viscosity_Pas',
        'source_roughness_rms',
        'source_roughness_corr_length_m',
        'source_roughness_slope_rms',
        'source_adhesion_energy_Jm2',
        'source_resuspension_roughness_scale',
        'source_resuspension_adhesion_scale',
        'source_resuspension_tau_roughness_scale',
        'source_resuspension_tau_adhesion_scale',
        'source_resuspension_tau_slope_scale',
    }.issubset(set(get_source_law('resuspension_shear_material').parameters))


def test_prepared_runtime_summary_reports_non_interpretive_model_inputs(tmp_path: Path):
    cfg_path = _write_minimal_2d_config(
        tmp_path / 'cfg_model_inputs',
        solver_updates={
            'drag_model': 'epstein',
            'stochastic_motion': {'enabled': True},
            'plasma_background': {
                'source': 'saas_constant',
                'electron_density_m3': 1.0e16,
                'ion_density_m3': 1.0e16,
                'electron_temperature_eV': 3.0,
                'ion_temperature_eV': 0.03,
                'ion_mass_amu': 69.0,
                'ion_charge_number': 1.0,
            },
            'charge_model': {
                'enabled': True,
                'mode': 'finite_rate_flux_balance',
            },
        },
    )

    summary = prepared_runtime_summary(build_prepared_runtime_from_yaml(cfg_path))
    model_inputs = summary['model_input_summary']

    assert model_inputs['drag_model'] == 'epstein'
    assert model_inputs['enabled_forces'] == ['drag', 'brownian']
    assert model_inputs['force_enabled_reason']['drag'] == 'required_solver'
    assert model_inputs['force_enabled_reason']['brownian'] == 'stochastic_motion_default'
    assert model_inputs['stochastic_motion_enabled'] == 1
    assert model_inputs['stochastic_motion_temperature_source'] == 'field_T_then_gas'
    assert model_inputs['charge_model_enabled'] == 1
    assert model_inputs['charge_model_mode'] == 'finite_rate_flux_balance'
    assert model_inputs['charge_background_source'] == 'plasma_background'
    assert model_inputs['charge_model_support_scope'] == '2d_regular_rectilinear_field_or_scalar_plasma_background'
    assert model_inputs['plasma_background_enabled'] == 1
    assert model_inputs['plasma_background_source'] == 'saas_constant'
    assert model_inputs['near_wall_force_correction_applied'] == 0
    assert model_inputs['particle_coupling'] == 'one_way'
    assert model_inputs['particle_wall_contact_geometry'] == 'center_position'
    assert model_inputs['source_law_usage'] == {'explicit_csv': 3}


def test_prepared_runtime_summary_uses_solver_parsers_for_model_aliases(tmp_path: Path):
    cfg_path = _write_minimal_2d_config(
        tmp_path / 'cfg_model_input_aliases',
        solver_updates={
            'stochastic_motion': {'enabled': 'yes', 'temperature_source': 'field_t_then_gas'},
            'plasma_background': {
                'source': 'constant',
                'electron_density_m3': 1.0e16,
                'ion_density_m3': 1.0e16,
                'electron_temperature_eV': 2.5,
                'ion_temperature_eV': 0.04,
                'ion_mass_amu': 69.0,
                'ion_charge_number': 1.0,
            },
            'charge_model': {
                'enabled': 'on',
                'mode': 'saas_flux_balance',
                'background_source': 'saas_constant',
            },
        },
    )

    model_inputs = prepared_runtime_summary(build_prepared_runtime_from_yaml(cfg_path))['model_input_summary']

    assert model_inputs['stochastic_motion_enabled'] == 1
    assert model_inputs['stochastic_motion_temperature_source'] == 'field_T_then_gas'
    assert model_inputs['charge_model_mode'] == 'finite_rate_flux_balance'
    assert model_inputs['charge_background_source'] == 'plasma_background'
    assert model_inputs['plasma_background_source'] == 'saas_constant'


def _single_particle_table() -> ParticleTable:
    return ParticleTable(
        spatial_dim=2,
        particle_id=np.asarray([1], dtype=np.int64),
        position=np.asarray([[0.0, 0.0]], dtype=np.float64),
        velocity=np.asarray([[0.0, 0.0]], dtype=np.float64),
        release_time=np.asarray([0.0], dtype=np.float64),
        mass=np.asarray([1.0e-12], dtype=np.float64),
        diameter=np.asarray([1.0e-6], dtype=np.float64),
        density=np.asarray([1000.0], dtype=np.float64),
        charge=np.asarray([0.0], dtype=np.float64),
        source_part_id=np.asarray([7], dtype=np.int64),
        material_id=np.asarray([3], dtype=np.int64),
        source_event_tag=np.asarray([''], dtype=object),
        source_law_override=np.asarray([''], dtype=object),
        source_speed_scale_override=np.asarray([np.nan], dtype=np.float64),
        stick_probability=np.asarray([np.nan], dtype=np.float64),
        dep_particle_rel_permittivity=np.asarray([np.nan], dtype=np.float64),
        thermophoretic_coeff=np.asarray([np.nan], dtype=np.float64),
    )


def test_boundary_release_failed_offset_is_preflight_error_when_input_check_off(tmp_path: Path):
    particles = _single_particle_table()
    diagnostics = (
        {
            'particle_id': 1,
            'source_part_id': 7,
            'input_material_id': 3,
            'resolved_material_id': 3,
            'law_name': 'explicit_csv',
            'release_enabled': 1,
            'particle_diameter_m': 1.0e-6,
            'source_delta_speed_mps': 0.0,
            'final_speed_mps': 0.0,
            'boundary_release_applied': 1,
            'boundary_release_inside_after_offset': 0,
            'boundary_release_part_id': 7,
            'boundary_release_primitive_id': 12,
            'boundary_release_distance_m': 0.0,
            'boundary_release_solver_offset_m': 1.0e-6,
            'boundary_release_total_offset_m': 1.0e-6,
            'source_position_offset_m': 0.0,
            'original_x': 0.0,
            'original_y': 0.0,
            'release_x': -1.0e-6,
            'release_y': 0.0,
        },
    )
    source_preprocess = SourcePreprocessResult(
        particles=particles,
        resolved=None,  # type: ignore[arg-type]
        source_model_summary={'boundary_release_failed_offset_count': 1},
        diagnostics_rows=diagnostics,
        release_enabled=np.asarray([True], dtype=bool),
    )
    runtime = RuntimeLike(
        spatial_dim=2,
        coordinate_system='cartesian_xy',
        particles=particles,
        walls=None,
        materials=None,
        source_events=None,
        process_steps=None,
        compiled_source_events=None,
        geometry_provider=None,
        field_provider=None,
        gas=GasProperties(),
        config_payload={'input_contract': {'initial_particle_field_support': 'off'}},
        source_preprocess=source_preprocess,
    )
    prepared = PreparedRuntime(runtime=runtime, source_preprocess=source_preprocess)
    out_dir = tmp_path / 'out_boundary_release_failure'

    with pytest.raises(ValueError, match='boundary_release offset produced positions outside'):
        enforce_initial_particle_field_support(prepared, out_dir)

    report = json.loads((out_dir / 'input_contract_report.json').read_text(encoding='utf-8'))
    assert report['passed'] is False
    assert int(report['boundary_release_failed_offset_count']) == 1
    assert int(report['boundary_release_offset_failures'][0]['particle_id']) == 1
    assert (out_dir / 'source_particle_diagnostics.csv').exists()


def test_flake_preprocess_does_not_require_unused_wall_shear_for_comsol_field() -> None:
    particles = _single_particle_table()
    axes = (np.asarray([0.0, 1.0], dtype=np.float64), np.asarray([0.0, 1.0], dtype=np.float64))
    times = np.asarray([0.0], dtype=np.float64)
    zeros = np.zeros((1, 2, 2), dtype=np.float64)
    field = RegularFieldND(
        spatial_dim=2,
        coordinate_system='cartesian_xy',
        axis_names=('x', 'y'),
        axes=axes,
        quantities={
            'ux': QuantitySeriesND('ux', '', times, zeros.copy()),
            'uy': QuantitySeriesND('uy', '', times, zeros.copy()),
        },
        valid_mask=np.ones((2, 2), dtype=bool),
        metadata={'source_kind': 'comsol_export_bundle_field'},
    )
    runtime = RuntimeLike(
        spatial_dim=2,
        coordinate_system='cartesian_xy',
        particles=particles,
        walls=None,
        materials=None,
        source_events=None,
        process_steps=None,
        compiled_source_events=None,
        geometry_provider=None,
        field_provider=FieldProviderND(field=field, kind='precomputed_npz'),
        gas=GasProperties(),
        config_payload={
            'source': {
                'default_law': 'flake_normal_escape_material',
                'preprocess': {'enabled': True},
            }
        },
    )

    prepared = prepare_runtime(runtime, seed=1)

    assert prepared.source_preprocess is not None
    assert prepared.source_preprocess.source_model_summary['law_usage'] == {'flake_normal_escape_material': 1}


def test_surface_release_preprocess_offsets_boundary_particle_and_reflects_velocity(tmp_path: Path):
    field_path = _write_invalid_left_field_bundle(
        tmp_path / 'field_surface_release.npz',
        invalid_until_x=-0.95,
    )
    particles_path = _write_particle_row(
        tmp_path / 'surface_release_particles.csv',
        _one_particle_row(spatial_dim=2, x=-1.0, y=0.0, vx=-0.2, vy=0.0),
    )
    cfg_path = _write_precomputed_field_config(
        tmp_path / 'cfg_surface_release',
        field_path=field_path,
        particles_path=particles_path,
        solver_updates={'dt': 0.005, 't_end': 0.01, 'save_every': 1, 'valid_mask_policy': 'retry_then_stop'},
        output_updates={'write_collision_diagnostics': 1, 'write_source_diagnostics': 1},
        input_mode='warn',
        provider_contract={'boundary_field_support': 'off'},
        source_preprocess_enabled=True,
    )
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    cfg.setdefault('source', {})['source_position_offset_m'] = 0.1
    cfg.setdefault('source', {}).setdefault('preprocess', {}).update(
        {
            'boundary_release': True,
            'normal_velocity_policy': 'reflect_inward',
        }
    )
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

    prepared = build_prepared_runtime_from_yaml(cfg_path)
    particles = prepared.runtime.particles
    assert prepared.source_preprocess is not None
    source_row = prepared.source_preprocess.diagnostics_rows[0]
    expected_release_x = -1.0 + float(source_row['boundary_release_total_offset_m'])

    assert particles.position[0, 0] == pytest.approx(expected_release_x)
    assert particles.position[0, 1] == pytest.approx(0.0)
    assert particles.velocity[0, 0] == pytest.approx(0.2)
    assert particles.velocity[0, 1] == pytest.approx(0.0)
    assert int(particles.source_part_id[0]) == 10
    assert int(source_row['source_part_id']) == 10
    assert source_row['source_provenance_group'] == 'production_generated_source'
    assert source_row['source_position_offset_m'] == pytest.approx(0.1)
    assert 0.0 < float(source_row['boundary_release_solver_offset_m']) < 1.0e-5
    assert source_row['boundary_release_capture_tolerance_m'] == pytest.approx(source_row['capture_tolerance_m'])
    assert source_row['boundary_release_inward_offset_m'] == pytest.approx(
        source_row['boundary_release_solver_offset_m']
    )
    assert source_row['projection_distance_m'] == pytest.approx(source_row['boundary_release_distance_m'])
    assert int(source_row['projected_part_id']) == int(source_row['boundary_release_part_id'])
    assert int(source_row['projected_boundary_id']) == int(source_row['boundary_release_primitive_id'])
    assert source_row['release_preprocess_mode'] == 'projected_inward_offset'
    assert source_row['raw_x'] == pytest.approx(-1.0)
    assert source_row['projected_x'] == pytest.approx(-1.0)
    assert source_row['initial_after_preprocess_x'] == pytest.approx(expected_release_x)
    assert source_row['boundary_release_total_offset_m'] == pytest.approx(
        float(source_row['source_position_offset_m']) + float(source_row['boundary_release_solver_offset_m'])
    )
    summary = prepared.source_preprocess.source_model_summary
    assert summary['boundary_release_applied_count'] == 1
    assert summary['boundary_release_projection_distance_count'] == 1
    assert summary['boundary_release_projection_distance_max_m'] == pytest.approx(0.0)
    assert summary['source_provenance_counts'] == {'production_generated_source': 1}

    input_report = build_initial_particle_field_support_report(prepared)
    assert int(input_report['status_counts']['hard_invalid']) == 0

    out_dir = tmp_path / 'out_surface_release'
    run_solver_2d_from_yaml(cfg_path, output_dir=out_dir)
    diagnostics = _collision_diagnostics(out_dir)
    assert int(diagnostics['invalid_mask_stopped_count']) == 0
    source_diag = _read_table(out_dir / 'source_particle_diagnostics.csv')
    assert source_diag.loc[0, 'original_x'] == pytest.approx(-1.0)
    assert source_diag.loc[0, 'raw_x'] == pytest.approx(-1.0)
    assert source_diag.loc[0, 'release_x'] == pytest.approx(expected_release_x)
    assert source_diag.loc[0, 'initial_after_preprocess_x'] == pytest.approx(expected_release_x)
    assert source_diag.loc[0, 'projection_distance_m'] == pytest.approx(0.0)
    assert int(source_diag.loc[0, 'boundary_release_applied']) == 1
    assert int(source_diag.loc[0, 'source_part_id']) == 10
    assert source_diag.loc[0, 'source_provenance_group'] == 'production_generated_source'


def test_surface_release_capture_tolerance_and_inward_offset_are_independent(tmp_path: Path):
    field_path = _write_invalid_left_field_bundle(
        tmp_path / 'field_surface_release_independent.npz',
        invalid_until_x=-0.95,
    )
    particles_path = _write_particle_row(
        tmp_path / 'surface_release_independent_particles.csv',
        _one_particle_row(spatial_dim=2, x=-1.0002, y=0.0, vx=0.0, vy=0.0),
    )
    cfg_path = _write_precomputed_field_config(
        tmp_path / 'cfg_surface_release_independent',
        field_path=field_path,
        particles_path=particles_path,
        input_mode='warn',
        provider_contract={'boundary_field_support': 'off'},
        source_preprocess_enabled=True,
    )
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    cfg.setdefault('source', {})['source_position_offset_m'] = 0.0
    cfg.setdefault('source', {}).setdefault('preprocess', {}).update(
        {
            'boundary_release': True,
            'boundary_capture_tolerance_m': 5.0e-4,
            'boundary_inward_offset_m': 2.0e-2,
        }
    )
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

    prepared = build_prepared_runtime_from_yaml(cfg_path)

    assert prepared.source_preprocess is not None
    row = prepared.source_preprocess.diagnostics_rows[0]
    summary = prepared.source_preprocess.source_model_summary
    assert int(row['boundary_release_applied']) == 1
    assert row['capture_tolerance_m'] == pytest.approx(5.0e-4)
    assert row['inward_offset_m'] == pytest.approx(2.0e-2)
    assert row['projection_distance_m'] == pytest.approx(2.0e-4)
    assert row['projected_x'] == pytest.approx(-1.0)
    assert row['initial_after_preprocess_x'] == pytest.approx(-0.98)
    assert prepared.runtime.particles.position[0, 0] == pytest.approx(-0.98)
    assert int(row['projected_part_id']) == 10
    assert int(row['projected_boundary_id']) >= 0
    assert summary['boundary_release_capture_tolerance_m'] == pytest.approx(5.0e-4)
    assert summary['boundary_release_inward_offset_m'] == pytest.approx(2.0e-2)
    assert summary['boundary_release_projection_distance_max_m'] == pytest.approx(2.0e-4)


def test_surface_release_large_capture_uses_small_default_inward_offset(tmp_path: Path):
    field_path = _write_invalid_left_field_bundle(
        tmp_path / 'field_surface_release_large_capture.npz',
        invalid_until_x=-0.95,
    )
    particles_path = _write_particle_row(
        tmp_path / 'surface_release_large_capture_particles.csv',
        _one_particle_row(spatial_dim=2, x=-1.0002, y=0.0, vx=0.0, vy=0.0),
    )
    cfg_path = _write_precomputed_field_config(
        tmp_path / 'cfg_surface_release_large_capture',
        field_path=field_path,
        particles_path=particles_path,
        solver_updates={'on_boundary_tol_m': 2.0e-6},
        input_mode='warn',
        provider_contract={'boundary_field_support': 'off'},
        source_preprocess_enabled=True,
    )
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    cfg.setdefault('wall', {})['epsilon_offset_m'] = 1.0e-6
    cfg.setdefault('source', {})['source_position_offset_m'] = 0.0
    cfg.setdefault('source', {}).setdefault('preprocess', {}).update(
        {
            'boundary_release': True,
            'boundary_capture_tolerance_m': 5.0e-4,
        }
    )
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

    prepared = build_prepared_runtime_from_yaml(cfg_path)

    assert prepared.source_preprocess is not None
    row = prepared.source_preprocess.diagnostics_rows[0]
    summary = prepared.source_preprocess.source_model_summary
    assert int(row['boundary_release_applied']) == 1
    assert row['capture_tolerance_m'] == pytest.approx(5.0e-4)
    assert row['inward_offset_m'] == pytest.approx(2.0e-6)
    assert row['boundary_release_solver_offset_m'] == pytest.approx(2.0e-6)
    assert row['projection_distance_m'] == pytest.approx(2.0e-4)
    assert row['projected_x'] == pytest.approx(-1.0)
    assert row['initial_after_preprocess_x'] == pytest.approx(-1.0 + 2.0e-6)
    assert prepared.runtime.particles.position[0, 0] == pytest.approx(-1.0 + 2.0e-6)
    assert summary['boundary_release_capture_tolerance_m'] == pytest.approx(5.0e-4)
    assert summary['boundary_release_inward_offset_m'] == pytest.approx(2.0e-6)
    assert summary['boundary_release_projection_distance_max_m'] == pytest.approx(2.0e-4)


def test_surface_release_capture_tolerance_controls_classification_before_offset(tmp_path: Path):
    field_path = _write_invalid_left_field_bundle(
        tmp_path / 'field_surface_release_capture_gate.npz',
        invalid_until_x=-0.95,
    )
    particles_path = _write_particle_row(
        tmp_path / 'surface_release_capture_gate_particles.csv',
        _one_particle_row(spatial_dim=2, x=-1.0002, y=0.0, vx=0.0, vy=0.0),
    )
    cfg_path = _write_precomputed_field_config(
        tmp_path / 'cfg_surface_release_capture_gate',
        field_path=field_path,
        particles_path=particles_path,
        input_mode='warn',
        provider_contract={'boundary_field_support': 'off'},
        source_preprocess_enabled=True,
    )
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    cfg.setdefault('source', {})['source_position_offset_m'] = 0.0
    cfg.setdefault('source', {}).setdefault('preprocess', {}).update(
        {
            'boundary_release': True,
            'boundary_capture_tolerance_m': 1.0e-5,
            'boundary_inward_offset_m': 2.0e-2,
        }
    )
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

    prepared = build_prepared_runtime_from_yaml(cfg_path)

    assert prepared.source_preprocess is not None
    row = prepared.source_preprocess.diagnostics_rows[0]
    assert int(row['boundary_release_applied']) == 0
    assert row['capture_tolerance_m'] == pytest.approx(1.0e-5)
    assert row['inward_offset_m'] == pytest.approx(2.0e-2)
    assert row['release_preprocess_mode'] == 'raw'
    assert prepared.runtime.particles.position[0, 0] == pytest.approx(-1.0002)

def test_unknown_source_provenance_is_not_classified_without_boundary_release(tmp_path: Path):
    field_path = _write_invalid_left_field_bundle(
        tmp_path / 'field_unknown_source_raw.npz',
        invalid_until_x=-0.95,
    )
    particle = _one_particle_row(spatial_dim=2, x=-1.0002, y=0.0, vx=0.0, vy=0.0)
    particle['source_part_id'] = 0
    particles_path = _write_particle_row(tmp_path / 'unknown_source_raw_particles.csv', particle)
    cfg_path = _write_precomputed_field_config(
        tmp_path / 'cfg_unknown_source_raw',
        field_path=field_path,
        particles_path=particles_path,
        input_mode='warn',
        provider_contract={'boundary_field_support': 'off'},
        source_preprocess_enabled=True,
    )

    prepared = build_prepared_runtime_from_yaml(cfg_path)

    assert prepared.source_preprocess is not None
    row = prepared.source_preprocess.diagnostics_rows[0]
    summary = prepared.source_preprocess.source_model_summary
    assert int(prepared.runtime.particles.source_part_id[0]) == 0
    assert prepared.runtime.particles.position[0, 0] == pytest.approx(-1.0002)
    assert int(row['source_part_id']) == 0
    assert int(row['boundary_release_applied']) == 0
    assert row['source_provenance_group'] == 'unknown_source'
    assert summary['boundary_release_applied_count'] == 0
    assert summary['source_provenance_counts'] == {'unknown_source': 1}


def test_unknown_source_can_boundary_release_only_when_explicitly_enabled(tmp_path: Path):
    field_path = _write_invalid_left_field_bundle(
        tmp_path / 'field_unknown_source_boundary_release.npz',
        invalid_until_x=-0.95,
    )
    particle = _one_particle_row(spatial_dim=2, x=-1.0002, y=0.0, vx=0.0, vy=0.0)
    particle['source_part_id'] = 0
    particles_path = _write_particle_row(tmp_path / 'unknown_source_boundary_release_particles.csv', particle)
    cfg_path = _write_precomputed_field_config(
        tmp_path / 'cfg_unknown_source_boundary_release',
        field_path=field_path,
        particles_path=particles_path,
        input_mode='warn',
        provider_contract={'boundary_field_support': 'off'},
        source_preprocess_enabled=True,
    )
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    cfg.setdefault('source', {})['source_position_offset_m'] = 0.0
    cfg.setdefault('source', {}).setdefault('preprocess', {}).update(
        {
            'boundary_release': True,
            'boundary_capture_tolerance_m': 5.0e-4,
            'boundary_inward_offset_m': 2.0e-6,
        }
    )
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

    prepared = build_prepared_runtime_from_yaml(cfg_path)

    assert prepared.source_preprocess is not None
    row = prepared.source_preprocess.diagnostics_rows[0]
    summary = prepared.source_preprocess.source_model_summary
    assert int(prepared.runtime.particles.source_part_id[0]) == 0
    assert int(row['source_part_id']) == 0
    assert int(row['boundary_release_applied']) == 1
    assert int(row['boundary_release_part_id']) == 10
    assert row['source_provenance_group'] == 'production_generated_source'
    assert prepared.runtime.particles.position[0, 0] == pytest.approx(-1.0 + 2.0e-6)
    assert summary['source_provenance_counts'] == {'production_generated_source': 1}

def test_surface_release_requires_explicit_boundary_primitives() -> None:
    runtime = SimpleNamespace(
        spatial_dim=2,
        geometry_provider=SimpleNamespace(geometry=SimpleNamespace(boundary_edges=None)),
    )

    with pytest.raises(ValueError, match='boundary_release requires explicit boundary primitives'):
        boundary_service_for_source_preprocess(runtime, 1.0e-6)

def test_surface_release_prepare_requires_boundary_primitives(tmp_path: Path) -> None:
    axes = np.linspace(-1.0, 1.0, 21)
    field_path = _write_field_bundle(tmp_path / 'field_no_release_primitives.npz', axes, axes)
    geometry_path = _write_precomputed_geometry_npz(
        tmp_path / 'geometry_no_release_primitives.npz',
        axes,
        axes,
        valid_mask=np.ones((axes.size, axes.size), dtype=bool),
    )
    cfg_path = _write_minimal_2d_config(
        tmp_path / 'cfg_no_release_primitives',
        provider_updates={
            'geometry': {'kind': 'precomputed_npz', 'npz_path': str(geometry_path.resolve())},
            'field': {'kind': 'precomputed_npz', 'npz_path': str(field_path.resolve())},
        },
    )
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    cfg.setdefault('source', {}).setdefault('preprocess', {})['boundary_release'] = True
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

    with pytest.raises(ValueError, match='boundary_release requires explicit boundary primitives'):
        build_prepared_runtime_from_yaml(cfg_path)

def test_surface_release_preprocess_does_not_rescue_far_outside_particle(tmp_path: Path):
    field_path = _write_invalid_left_field_bundle(
        tmp_path / 'field_surface_release_far_outside.npz',
        invalid_until_x=-0.95,
    )
    particles_path = _write_particle_row(
        tmp_path / 'surface_release_far_outside_particles.csv',
        _one_particle_row(spatial_dim=2, x=-1.5, y=0.0, vx=0.0, vy=0.0),
    )
    cfg_path = _write_precomputed_field_config(
        tmp_path / 'cfg_surface_release_far_outside',
        field_path=field_path,
        particles_path=particles_path,
        input_mode='warn',
        provider_contract={'boundary_field_support': 'off'},
        source_preprocess_enabled=True,
    )
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    cfg.setdefault('source', {})['source_position_offset_m'] = 0.1
    cfg.setdefault('source', {}).setdefault('preprocess', {}).update(
        {
            'boundary_release': True,
            'normal_velocity_policy': 'reflect_inward',
        }
    )
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

    prepared = build_prepared_runtime_from_yaml(cfg_path)
    report = build_initial_particle_field_support_report(prepared)

    assert prepared.source_preprocess is not None
    assert prepared.source_preprocess.source_model_summary['boundary_release_applied_count'] == 0
    assert int(report['status_counts']['hard_invalid']) == 1

def test_surface_release_rejects_object_form_and_legacy_aliases(tmp_path: Path):
    field_path = _write_invalid_left_field_bundle(
        tmp_path / 'field_surface_release_legacy_alias.npz',
        invalid_until_x=-0.95,
    )
    particles_path = _write_particle_row(
        tmp_path / 'surface_release_legacy_alias_particles.csv',
        _one_particle_row(spatial_dim=2, x=-1.0, y=0.0, vx=0.0, vy=0.0),
    )
    cfg_path = _write_precomputed_field_config(
        tmp_path / 'cfg_surface_release_legacy_alias',
        field_path=field_path,
        particles_path=particles_path,
        input_mode='warn',
        provider_contract={'boundary_field_support': 'off'},
        source_preprocess_enabled=True,
    )

    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    preprocess = cfg.setdefault('source', {}).setdefault('preprocess', {})
    preprocess['boundary_release'] = {'enabled': True}
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    with pytest.raises(ValueError, match='boundary_release must be true or false'):
        build_prepared_runtime_from_yaml(cfg_path)

    preprocess['boundary_release'] = False
    preprocess[f'boundary_release_{"enabled"}'] = True
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    with pytest.raises(ValueError, match='legacy boundary-release alias'):
        build_prepared_runtime_from_yaml(cfg_path)

def test_class_match_ratio_compares_particle_end_states_by_particle_id():
    reference = _state_frame([(1, 'active_free_flight'), (2, 'stuck'), (3, 'absorbed'), (4, 'escaped')])
    candidate = _state_frame([(4, 'escaped'), (3, 'absorbed'), (2, 'active_free_flight'), (1, 'active_free_flight')])
    ratio, compared = class_match_ratio(candidate, reference)
    assert compared == 4
    assert ratio == pytest.approx(0.75)

def test_class_match_ratio_uses_shared_particle_ids_only():
    reference = _state_frame([(1, 'active_free_flight'), (2, 'stuck')])
    candidate = _state_frame([(2, 'stuck'), (3, 'escaped')])
    ratio, compared = class_match_ratio(candidate, reference)
    assert compared == 1
    assert ratio == pytest.approx(1.0)

def test_class_match_ratio_recognizes_invalid_mask_stopped_as_distinct_class():
    reference = _state_frame([(1, 'invalid_mask_stopped'), (2, 'active_free_flight')])
    candidate = _state_frame([(1, 'active_free_flight'), (2, 'active_free_flight')])
    ratio, compared = class_match_ratio(candidate, reference)
    assert compared == 2
    assert ratio == pytest.approx(0.5)

def test_class_transition_summary_reports_mismatched_end_states():
    reference = _state_frame([(1, 'stuck'), (2, 'invalid_mask_stopped'), (3, 'active_free_flight')])
    candidate = _state_frame([(1, 'invalid_mask_stopped'), (2, 'stuck'), (3, 'active_free_flight')])
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
            '--reference-scope',
            'sampled',
        ]
    )
    assert rc == 0
    summary_files = sorted(output_root.glob('compare_*/comparison_summary.json'))
    assert summary_files
    stable_summary = output_root / 'comparison_summary.json'
    assert stable_summary.exists()
    summary = json.loads(summary_files[-1].read_text(encoding='utf-8'))
    stable = json.loads(stable_summary.read_text(encoding='utf-8'))
    assert stable['comparison_dir'] == summary['comparison_dir']
    assert summary['summary_schema_version'] == 1
    assert summary['reference_scope'] == 'sampled'
    assert summary['reference']['coordinate_system'] == 'cartesian_xy'
    assert summary['runs'][0]['coordinate_system'] == 'cartesian_xy'
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
