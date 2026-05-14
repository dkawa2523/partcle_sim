from __future__ import annotations

import regression_test as _regression_helpers

globals().update({
    name: value
    for name, value in vars(_regression_helpers).items()
    if not name.startswith("__")
})

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

def test_triangle_mesh_field_backend_runs_in_2d_solver(tmp_path: Path):
    mesh_field_path = _write_triangle_mesh_field_npz(tmp_path / 'field_mesh.npz')
    particles_path = _write_rows_csv(
        tmp_path / 'particles_mesh.csv',
        [_one_particle_row(spatial_dim=2, x=0.25, y=0.25, vx=0.0, diameter=1e-6, density=1200.0)],
    )
    config_path = _write_minimal_2d_config(
        tmp_path,
        path_updates={'particles_csv': str(particles_path.resolve())},
        geometry_updates={'bounds': [0.0, 1.0, 0.0, 1.0], 'grid_shape': [41, 41]},
        provider_updates={'field': {'kind': 'precomputed_triangle_mesh_npz', 'npz_path': str(mesh_field_path.resolve())}},
        solver_updates={
            'dt': 0.05,
            't_end': 0.05,
            'save_every': 1,
            'integrator': 'etd2',
            'field_backend_mode': 'triangle_mesh',
            'forces': {'pressure_gradient': {'enabled': True}},
        },
        output_updates={'write_collision_diagnostics': 1},
    )
    out_dir = tmp_path / 'out_mesh_backend'
    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    report = _solver_report(out_dir)
    final_df = _final_particles(out_dir)
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
    summary = _read_json(out_dir / 'generated' / 'comsol_case_summary.json')
    assert summary['provider_contract']['passed'] is True
    assert (out_dir / 'generated' / 'provider_contract_report.json').exists()
    assert summary['field_summary']['geometry_mask_applied'] is False
    assert int(summary['field_summary']['field_ghost_cells']) == 0
    assert not (out_dir / 'generated' / 'comsol_field_mesh_2d.npz').exists()
    assert not (out_dir / 'run_config_mesh.yaml').exists()

def test_comsol_builder_particles_only_generates_boundary_release_sources(tmp_path: Path):
    out_dir = tmp_path / 'comsol_case_boundary_release_particles'
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
    )
    particles = pd.read_csv(out_dir / 'particles.csv')
    assert len(particles) == 128
    assert particles['release_time'].iloc[0] == pytest.approx(0.0)
    assert particles['release_time'].iloc[-1] == pytest.approx(0.4)
    assert float(particles['release_offset_m'].max()) == pytest.approx(0.0)
    assert np.allclose(particles['x'].to_numpy(dtype=float), particles['source_x'].to_numpy(dtype=float))
    assert np.allclose(particles['y'].to_numpy(dtype=float), particles['source_y'].to_numpy(dtype=float))
    cfg = yaml.safe_load((out_dir / 'run_config.yaml').read_text(encoding='utf-8'))
    assert cfg['source']['source_position_offset_m'] == pytest.approx(0.0)
    assert cfg['source']['preprocess']['enabled'] is True
    assert cfg['source']['preprocess']['boundary_release'] is True
    prepared = build_prepared_runtime_from_yaml(out_dir / 'run_config.yaml')
    assert prepared.source_preprocess is not None
    assert prepared.source_preprocess.source_model_summary['boundary_release_applied_count'] == 128
    assert prepared.source_preprocess.source_model_summary['boundary_release_failed_offset_count'] == 0
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
    _run_minimal_case(2, out_dir)

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
    _run_minimal_case(3, out_dir)
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
    config_path = _write_wall_bounce_config(
        tmp_path,
        spatial_dim=2,
        name='fast',
        vx=50.0,
        max_wall_hits_per_step=1,
    )
    out_dir = tmp_path / 'out_diag_hits'
    run_solver_2d_from_yaml(config_path, output_dir=out_dir)

    diag_path = out_dir / 'collision_diagnostics.json'
    assert diag_path.exists()
    diag = _read_json(diag_path)
    assert int(diag['max_wall_hits_per_step']) == 1
    assert int(diag['max_hits_reached_count']) >= 1
    assert int(diag['multi_hit_events_count']) == 0
    assert int(diag['max_hit_event_summary']['event_count']) == int(diag['max_hits_reached_count'])
    assert float(diag['max_hit_event_summary']['remaining_dt_total_s']) >= 0.0
    boundary_diag = diag['boundary_diagnostics']
    assert boundary_diag['wall_law_semantics']['pass_through'] == 'non_colliding_boundary'
    assert boundary_diag['wall_law_semantics']['open'] == 'particle_exit'
    assert boundary_diag['collision_boundary_geometry'] == 'linear_segment_or_triangle_boundary'
    assert int(boundary_diag['ambiguous_hit_count']) >= 0
    max_hit_events = _read_table(out_dir / 'max_hit_events.csv')
    assert not max_hit_events.empty
    assert set(max_hit_events.columns).issuperset({'time_s', 'particle_id', 'hits_in_step', 'remaining_dt_s', 'part_id_sequence'})
    assert int(max_hit_events['hits_in_step'].max()) >= 1

@pytest.mark.parametrize(('spatial_dim', 'vx'), [(2, 50.0), (3, 80.0)])
def test_collision_reintegration_counter_is_nonzero_for_wall_bounces(
    tmp_path: Path,
    spatial_dim: int,
    vx: float,
):
    config_path = _write_wall_bounce_config(
        tmp_path,
        spatial_dim=spatial_dim,
        name='reintegrate',
        vx=vx,
        max_wall_hits_per_step=3,
    )
    out_dir = tmp_path / f'out_diag_reintegrate_{spatial_dim}d'
    _run_solver_for_dim(spatial_dim, config_path, out_dir)
    diag = _collision_diagnostics(out_dir)
    assert int(diag['collision_reintegrated_segments_count']) >= 1

def test_adaptive_substep_diagnostics_toggle(tmp_path: Path):
    particles_path = _write_particle_row(
        tmp_path / 'adaptive_particles_2d.csv',
        _one_particle_row(spatial_dim=2, vx=5.0, diameter=1.0e-5, density=1200.0),
    )

    def write_adaptive_config(config_dir: Path, enabled: int) -> Path:
        return _write_particle_solver_config(
            config_dir,
            spatial_dim=2,
            particles_path=particles_path,
            geometry_updates={'bounds': [-10.0, 10.0, -10.0, 10.0], 'grid_shape': [51, 51]},
            field_updates={'shear_rate': 0.0},
            solver_updates={
                'dt': 0.1,
                't_end': 0.2,
                'save_every': 1,
                'min_tau_p_s': 1.0e-8,
                'integrator': 'etd',
                'adaptive_substep_enabled': int(enabled),
                'adaptive_substep_tau_ratio': 0.5,
                'adaptive_substep_max_splits': 4,
            },
            output_updates={'write_collision_diagnostics': 1},
        )

    cfg_off = write_adaptive_config(tmp_path / 'adaptive_off', 0)
    cfg_on = write_adaptive_config(tmp_path / 'adaptive_on', 1)
    out_off = tmp_path / 'out_adaptive_off'
    out_on = tmp_path / 'out_adaptive_on'
    run_solver_2d_from_yaml(cfg_off, output_dir=out_off)
    run_solver_2d_from_yaml(cfg_on, output_dir=out_on)
    diag_off = _collision_diagnostics(out_off)
    diag_on = _collision_diagnostics(out_on)
    assert int(diag_off['adaptive_substep_segments_count']) == 0
    assert int(diag_off['adaptive_substep_trigger_count']) == 0
    assert int(diag_on['adaptive_substep_enabled']) == 1
    assert int(diag_on['adaptive_substep_segments_count']) > 0
    assert int(diag_on['adaptive_substep_trigger_count']) > 0

def test_default_max_wall_hits_per_step_is_5(tmp_path: Path):
    out_dir = tmp_path / 'out_default_hits'
    _run_minimal_case(2, out_dir)
    diag = _collision_diagnostics(out_dir)
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
