from __future__ import annotations

import regression_helpers as _regression_helpers

from particle_tracer_unified.solvers.output_buffers import RuntimeBuffers
from particle_tracer_unified.solvers.runtime_plan import OUTPUT_MODE_MINIMAL, OutputPlan

globals().update({
    name: value
    for name, value in vars(_regression_helpers).items()
    if not name.startswith("__")
})

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

def test_minimal_runtime_buffers_do_not_allocate_step_summary_by_default():
    buffers = RuntimeBuffers(OutputPlan(mode=OUTPUT_MODE_MINIMAL, write_step_summary=False))

    assert buffers.step_summary is None
    assert buffers.summary() == {
        'output_mode': OUTPUT_MODE_MINIMAL,
        'output_minimal_enabled': 1,
        'output_debug_enabled': 0,
        'step_summary_buffer_enabled': 0,
    }

    explicit = RuntimeBuffers(OutputPlan(mode=OUTPUT_MODE_MINIMAL, write_step_summary=True))
    assert explicit.step_summary is not None
    assert explicit.summary()['step_summary_count'] == 0

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
    _write_particle_row(
        particles_path,
        _one_particle_row(spatial_dim=2, x=-0.8, y=-0.2, vx=0.2, diameter=1e-6, density=1200),
    )

    cfg = _write_precomputed_field_config(
        tmp_path / 'cfg_visual_invalid_stop',
        field_path=field_path,
        particles_path=particles_path,
        solver_updates={
            't_end': 0.2,
            'dt': 0.2,
            'save_every': 1,
            'integrator': 'etd2',
            'adaptive_substep_max_splits': 4,
            'valid_mask_policy': 'retry_then_stop',
        },
        output_updates={'mode': 'debug'},
        provider_contract={'boundary_field_support': 'off'},
    )

    out_dir = tmp_path / 'out_visual_invalid_stop'
    run_solver_2d_from_yaml(cfg, output_dir=out_dir)
    export_result_graphs(out_dir, case_dir=ROOT / 'examples' / 'minimal_2d', sample_trajectories=1)

    summary = json.loads((out_dir / 'visualizations' / 'graphs' / 'graph_summary.json').read_text(encoding='utf-8'))
    assert int(summary['final_state_counts']['invalid_mask_stopped']) == 1
    assert (out_dir / 'visualizations' / 'graphs' / '02_final_state_bar_and_pie.png').exists()

def test_export_boundary_diagnostics_reports_mixed_and_hard_invalid_regions(tmp_path: Path):
    case_dir, output_dir = _write_boundary_diagnostics_case(tmp_path)
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
    runtime = _square_boundary_runtime()

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
