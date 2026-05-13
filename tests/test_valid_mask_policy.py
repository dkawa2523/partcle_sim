from __future__ import annotations

import regression_test as _regression_helpers

globals().update({
    name: value
    for name, value in vars(_regression_helpers).items()
    if not name.startswith("__")
})

def test_initial_particle_field_support_contract_rejects_non_clean_start(tmp_path: Path):
    field_path = _write_invalid_left_field_bundle(
        tmp_path / 'field_initial_contract.npz',
        invalid_until_x=-0.75,
    )
    particles_path = _write_contract_particle(
        tmp_path / 'particles_initial_contract.csv',
        release_time=0.25,
    )
    cfg = _write_precomputed_field_config(
        tmp_path / 'cfg_initial_contract',
        field_path=field_path,
        particles_path=particles_path,
        provider_contract={'boundary_field_support': 'off'},
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
    field_path = _write_invalid_left_field_bundle(
        tmp_path / 'field_check_input_mode.npz',
        invalid_until_x=-0.75,
    )
    particles_path = _write_contract_particle(
        tmp_path / 'particles_check_input_mode.csv',
        release_time=0.0,
    )

    def _config_for(mode: str, cfg_dir: Path) -> Path:
        return _write_precomputed_field_config(
            cfg_dir,
            field_path=field_path,
            particles_path=particles_path,
            input_mode=mode,
            provider_contract={'boundary_field_support': 'off'},
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
    field_path = _write_invalid_left_field_bundle(
        tmp_path / 'field_boundary_contract.npz',
        invalid_until_x=-0.95,
    )
    cfg = _write_precomputed_field_config(
        tmp_path,
        field_path=field_path,
        provider_contract={
            'boundary_field_support': 'strict',
            'boundary_offset_cells': 0.5,
            'max_boundary_samples': 400,
        },
        source_preprocess_enabled=False,
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
    field_path = _write_transient_invalid_left_field_bundle(
        tmp_path / 'field_boundary_transient_contract.npz',
        invalid_until_x=-0.95,
    )

    cfg = _write_precomputed_field_config(
        tmp_path,
        field_path=field_path,
        provider_contract={
            'boundary_field_support': 'strict',
            'boundary_offset_cells': 0.5,
            'max_boundary_samples': 100,
            'max_time_samples': 2,
        },
        source_preprocess_enabled=False,
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
    config_path = _write_minimal_2d_config(
        tmp_path,
        solver_updates={'integrator': 'euler'},
    )
    with pytest.raises(ValueError, match='Unsupported solver.integrator'):
        build_prepared_runtime_from_yaml(config_path)

def test_eit_integrator_alias_is_rejected(tmp_path: Path):
    config_path = _write_minimal_2d_config(
        tmp_path,
        solver_updates={'integrator': 'eit'},
    )
    with pytest.raises(ValueError, match='Unsupported solver.integrator'):
        build_prepared_runtime_from_yaml(config_path)

def test_contact_tangent_motion_config_is_rejected(tmp_path: Path):
    config_path = _write_minimal_2d_config(
        tmp_path,
        solver_updates={'contact_tangent_motion': True},
    )

    with pytest.raises(ValueError, match='contact_tangent_motion is obsolete'):
        run_solver_2d_from_yaml(config_path, output_dir=tmp_path / 'out_contact_tangent_rejected')

def test_etd_integrator_runs_in_2d(tmp_path: Path):
    out_dir = tmp_path / 'out_2d_etd'
    config_path = _write_minimal_2d_config(
        tmp_path,
        solver_updates={'integrator': 'etd', 't_end': 0.1, 'save_every': 1},
    )
    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    report = _solver_report(out_dir)
    diag = _collision_diagnostics(out_dir)
    assert report['integrator'] == 'etd'
    assert diag['integrator'] == 'etd'

def test_etd2_integrator_runs_in_2d_and_3d(tmp_path: Path):
    out_2d = tmp_path / 'out_2d_etd2'
    cfg_2d = _write_minimal_config(
        tmp_path / 'cfg_2d_etd2',
        spatial_dim=2,
        solver_updates={'integrator': 'etd2', 't_end': 0.1, 'save_every': 1},
    )
    run_solver_2d_from_yaml(cfg_2d, output_dir=out_2d)
    report_2d = _solver_report(out_2d)
    diag_2d = _collision_diagnostics(out_2d)
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
    cfg_3d = _write_minimal_config(
        tmp_path / 'cfg_3d_etd2',
        spatial_dim=3,
        solver_updates={'integrator': 'etd2', 't_end': 0.1, 'save_every': 1},
    )
    run_solver_3d_from_yaml(cfg_3d, output_dir=out_3d)
    report_3d = _solver_report(out_3d)
    diag_3d = _collision_diagnostics(out_3d)
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

def test_stochastic_motion_disabled_preserves_solver_outputs(tmp_path: Path):
    cfg_base = _write_stochastic_motion_config(tmp_path / 'cfg_base', solver_seed=2468)
    cfg_disabled = _write_stochastic_motion_config(
        tmp_path / 'cfg_disabled',
        solver_seed=2468,
        stochastic_motion={'enabled': False},
    )
    out_base = tmp_path / 'out_base'
    out_disabled = tmp_path / 'out_disabled'
    run_solver_2d_from_yaml(cfg_base, output_dir=out_base)
    run_solver_2d_from_yaml(cfg_disabled, output_dir=out_disabled)
    np.testing.assert_allclose(_final_xy_velocity(out_base), _final_xy_velocity(out_disabled), rtol=0.0, atol=0.0)
    report = _solver_report(out_disabled)
    assert report['stochastic_motion']['enabled'] == 0

def test_stochastic_motion_seed_is_reproducible_and_changes_trajectory(tmp_path: Path):
    def stochastic(seed: int) -> dict[str, Any]:
        return {
            'enabled': True,
            'model': 'underdamped_langevin',
            'stride': 1,
            'seed': int(seed),
            'temperature_source': 'gas',
        }

    cfg_a = _write_stochastic_motion_config(tmp_path / 'cfg_a', stochastic_motion=stochastic(77))
    cfg_b = _write_stochastic_motion_config(tmp_path / 'cfg_b', stochastic_motion=stochastic(77))
    cfg_c = _write_stochastic_motion_config(tmp_path / 'cfg_c', stochastic_motion=stochastic(78))
    out_a = tmp_path / 'out_a'
    out_b = tmp_path / 'out_b'
    out_c = tmp_path / 'out_c'
    run_solver_2d_from_yaml(cfg_a, output_dir=out_a)
    run_solver_2d_from_yaml(cfg_b, output_dir=out_b)
    run_solver_2d_from_yaml(cfg_c, output_dir=out_c)

    np.testing.assert_allclose(_final_xy_velocity(out_a), _final_xy_velocity(out_b), rtol=0.0, atol=0.0)
    assert not np.allclose(_final_xy_velocity(out_a), _final_xy_velocity(out_c), rtol=0.0, atol=1.0e-12)
    report = _solver_report(out_a)
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

    def write_config(config_dir: Path, npz_path: Path) -> Path:
        return _write_minimal_2d_config(
            config_dir,
            provider_updates={
                'field': {
                    'kind': 'precomputed_npz',
                    'npz_path': str(npz_path.resolve()),
                },
            },
            solver_updates={'integrator': 'etd2', 't_end': 0.12, 'save_every': 1, 'valid_mask_policy': 'diagnostic'},
            input_contract_updates={'initial_particle_field_support': 'warn'},
            provider_contract_updates={'boundary_field_support': 'off'},
        )

    cfg_base = write_config(tmp_path / 'cfg_base', base_field_path)
    cfg_masked = write_config(tmp_path / 'cfg_masked', masked_field_path)
    out_base = tmp_path / 'out_base'
    out_masked = tmp_path / 'out_masked'
    run_solver_2d_from_yaml(cfg_base, output_dir=out_base)
    run_solver_2d_from_yaml(cfg_masked, output_dir=out_masked)

    base_final = _final_particles(out_base, sort=True)
    masked_final = _final_particles(out_masked, sort=True)
    assert base_final[['particle_id', 'active', 'stuck', 'absorbed', 'escaped']].equals(
        masked_final[['particle_id', 'active', 'stuck', 'absorbed', 'escaped']]
    )
    assert 'invalid_mask_stopped' in masked_final.columns
    assert int(masked_final['invalid_mask_stopped'].sum()) == 0
    for col in ('x', 'y', 'v_x', 'v_y'):
        assert masked_final[col].to_numpy() == pytest.approx(base_final[col].to_numpy(), abs=1.0e-12)

    base_report = _solver_report(out_base)
    masked_report = _solver_report(out_masked)
    masked_steps = _read_table(out_masked / 'runtime_step_summary.csv')
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
        collision_diagnostics=initial_collision_diagnostics(),
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

    particles_path = _write_particle_row(
        tmp_path / 'particles_single.csv',
        _one_particle_row(spatial_dim=2, x=-0.8, y=-0.2, vx=0.2, diameter=1e-6, density=1200),
    )
    cfg = _write_precomputed_field_config(
        tmp_path / 'cfg_retry_then_stop',
        field_path=field_path,
        particles_path=particles_path,
        solver_updates={'t_end': 0.2, 'dt': 0.2, 'save_every': 1, 'integrator': 'etd2', 'adaptive_substep_max_splits': 4},
        provider_contract={'boundary_field_support': 'off'},
    )

    out_dir = tmp_path / 'out_retry_then_stop'
    run_solver_2d_from_yaml(cfg, output_dir=out_dir)

    final_df = _final_particles(out_dir)
    report = _solver_report(out_dir)
    diag = _collision_diagnostics(out_dir)
    step_df = _read_table(out_dir / 'runtime_step_summary.csv')
    wall_df = _read_table(out_dir / 'wall_events.csv')

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

    particles_path = _write_particle_row(
        tmp_path / 'particles_single_invalid_start.csv',
        _one_particle_row(spatial_dim=2, x=-0.8, y=-0.2, vx=0.2, diameter=1e-6, density=1200),
    )
    cfg = _write_precomputed_field_config(
        tmp_path / 'cfg_retry_exhausted',
        field_path=field_path,
        particles_path=particles_path,
        solver_updates={
            't_end': 0.12,
            'dt': 0.12,
            'save_every': 1,
            'adaptive_substep_max_splits': 4,
            'valid_mask_policy': 'retry_then_stop',
        },
        input_mode='warn',
        provider_contract={'boundary_field_support': 'off'},
    )

    out_dir = tmp_path / 'out_retry_exhausted'
    run_solver_2d_from_yaml(cfg, output_dir=out_dir)

    final_df = _final_particles(out_dir)
    report = _solver_report(out_dir)
    diag = _collision_diagnostics(out_dir)
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
    diag = initial_collision_diagnostics()
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
    diag = initial_collision_diagnostics()
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
    diag = initial_collision_diagnostics()
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
