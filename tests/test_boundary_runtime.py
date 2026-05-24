from __future__ import annotations

import regression_helpers as _regression_helpers

globals().update({
    name: value
    for name, value in vars(_regression_helpers).items()
    if not name.startswith("__")
})

def test_batch_boundary_edge_hits_match_scalar_polyline_contract():
    runtime = _square_boundary_runtime()
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
    runtime = _square_boundary_runtime()

    part_ids, distances = nearest_boundary_edge_features_2d(
        runtime,
        np.asarray([[0.95, 0.5], [0.5, 0.1]], dtype=np.float64),
    )

    assert part_ids.tolist() == [2, 1]
    assert distances[0] == pytest.approx(0.05)
    assert distances[1] == pytest.approx(0.1)

def test_boundary_service_2d_matches_loop_truth_and_edge_hit_contract():
    runtime = _square_boundary_service_runtime()
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

def test_boundary_service_release_point_uses_source_part_hint_at_corner():
    runtime = _square_boundary_service_runtime(part_ids=(1, 2, 3, 4))
    service = build_boundary_service(runtime, spatial_dim=2, on_boundary_tol_m=1.0e-9, triangle_surface_3d=None)

    bottom = service.release_point(
        np.asarray([0.0, 0.0], dtype=np.float64),
        1,
        1.0e-3,
        1.0e-6,
    )
    left = service.release_point(
        np.asarray([0.0, 0.0], dtype=np.float64),
        4,
        1.0e-3,
        1.0e-6,
    )

    assert bool(bottom.is_on_boundary) is True
    assert int(bottom.nearest_part_id) == 1
    assert bottom.normal == pytest.approx([0.0, 1.0])
    assert bottom.offset_position == pytest.approx([0.0, 1.0e-3])
    assert bool(bottom.inside_after_offset) is True

    assert bool(left.is_on_boundary) is True
    assert int(left.nearest_part_id) == 4
    assert left.normal == pytest.approx([1.0, 0.0])
    assert left.offset_position == pytest.approx([1.0e-3, 0.0])
    assert bool(left.inside_after_offset) is True

def test_boundary_service_release_point_orients_3d_triangle_normal_inside():
    triangles = _cube_triangles_oriented()
    part_ids = np.ones(triangles.shape[0], dtype=np.int32)
    geometry = SimpleNamespace(
        spatial_dim=3,
        axes=tuple(np.asarray([-1.0, 0.0, 1.0], dtype=np.float64) for _ in range(3)),
        boundary_edges=None,
        boundary_loops_2d=(),
        boundary_triangles=triangles,
        boundary_triangle_part_ids=part_ids,
    )
    runtime = SimpleNamespace(geometry_provider=SimpleNamespace(geometry=geometry))
    surface = build_triangle_surface(triangles, part_ids, validate_closed=True)
    service = build_boundary_service(runtime, spatial_dim=3, on_boundary_tol_m=1.0e-7, triangle_surface_3d=surface)

    release = service.release_point(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        1,
        1.0e-3,
        1.0e-6,
    )

    assert bool(release.is_on_boundary) is True
    assert release.normal == pytest.approx([-1.0, 0.0, 0.0])
    assert release.offset_position == pytest.approx([0.999, 0.0, 0.0])
    assert bool(release.inside_after_offset) is True


def test_boundary_service_release_point_marks_3d_edge_and_vertex_ambiguous():
    triangles = _cube_triangles_oriented()
    part_ids = np.ones(triangles.shape[0], dtype=np.int32)
    geometry = SimpleNamespace(
        spatial_dim=3,
        axes=tuple(np.asarray([-1.0, 0.0, 1.0], dtype=np.float64) for _ in range(3)),
        boundary_edges=None,
        boundary_loops_2d=(),
        boundary_triangles=triangles,
        boundary_triangle_part_ids=part_ids,
    )
    runtime = SimpleNamespace(geometry_provider=SimpleNamespace(geometry=geometry))
    surface = build_triangle_surface(triangles, part_ids, validate_closed=True)
    service = build_boundary_service(runtime, spatial_dim=3, on_boundary_tol_m=1.0e-7, triangle_surface_3d=surface)

    edge_release = service.release_point(
        np.asarray([1.0, 1.0, 0.0], dtype=np.float64),
        1,
        1.0e-3,
        1.0e-6,
    )
    vertex_release = service.release_point(
        np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        1,
        1.0e-3,
        1.0e-6,
    )

    assert bool(edge_release.is_on_boundary) is True
    assert bool(edge_release.ambiguous) is True
    assert bool(vertex_release.is_on_boundary) is True
    assert bool(vertex_release.ambiguous) is True


@pytest.mark.parametrize(
    ('spatial_dim', 'bounds', 'grid_shape'),
    [
        (2, [-10.0, 10.0, -10.0, 10.0], [51, 51]),
        (3, [-10.0, 10.0, -10.0, 10.0, -10.0, 10.0], [31, 31, 31]),
    ],
)
def test_etd_reduces_position_error_for_linear_drag(
    tmp_path: Path,
    spatial_dim: int,
    bounds: list[float],
    grid_shape: list[int],
):
    particles_path = _write_particle_row(
        tmp_path / f'one_particle_{spatial_dim}d.csv',
        _one_particle_row(spatial_dim=spatial_dim),
    )
    drag_cfg = _write_integrator_config(
        tmp_path / f'drag_cfg_{spatial_dim}d',
        spatial_dim=spatial_dim,
        particles_path=particles_path,
        integrator='drag_relaxation',
        dt=0.05,
        bounds=bounds,
        grid_shape=grid_shape,
        shear_rate=0.0,
    )
    etd_cfg = _write_integrator_config(
        tmp_path / f'etd_cfg_{spatial_dim}d',
        spatial_dim=spatial_dim,
        particles_path=particles_path,
        integrator='etd',
        dt=0.05,
        bounds=bounds,
        grid_shape=grid_shape,
        shear_rate=0.0,
    )
    drag_out = tmp_path / f'out_drag_{spatial_dim}d'
    etd_out = tmp_path / f'out_etd_{spatial_dim}d'
    _run_solver_for_dim(spatial_dim, drag_cfg, drag_out)
    _run_solver_for_dim(spatial_dim, etd_cfg, etd_out)

    tau = 1000.0 * (1.0e-4 ** 2) / (18.0 * 1.8e-5)
    exact_x = tau * (1.0 - np.exp(-0.2 / tau))
    x_drag = float(_final_particles(drag_out).loc[0, 'x'])
    x_etd = float(_final_particles(etd_out).loc[0, 'x'])
    assert abs(x_etd - exact_x) < abs(x_drag - exact_x)

@pytest.mark.parametrize(
    ('spatial_dim', 'z0', 'bounds', 'grid_shape', 'shear_rate'),
    [
        (2, 0.0, [-10.0, 10.0, -10.0, 10.0], [81, 81], 4.0),
        (3, -0.1, [-10.0, 10.0, -10.0, 10.0, -10.0, 10.0], [41, 41, 41], 5.0),
    ],
)
def test_etd2_is_not_worse_than_etd_vs_fine_reference(
    tmp_path: Path,
    spatial_dim: int,
    z0: float,
    bounds: list[float],
    grid_shape: list[int],
    shear_rate: float,
):
    particles_path = _write_particle_row(
        tmp_path / f'one_particle_etd2_{spatial_dim}d.csv',
        _one_particle_row(spatial_dim=spatial_dim, y=0.2, z=z0, vx=0.8, diameter=1.0e-5),
    )
    cfg_dir = tmp_path / f'etd2_compare_{spatial_dim}d'
    cfg_etd = _write_integrator_config(
        cfg_dir / 'etd',
        spatial_dim=spatial_dim,
        particles_path=particles_path,
        integrator='etd',
        dt=0.05,
        bounds=bounds,
        grid_shape=grid_shape,
        shear_rate=shear_rate,
    )
    cfg_etd2 = _write_integrator_config(
        cfg_dir / 'etd2',
        spatial_dim=spatial_dim,
        particles_path=particles_path,
        integrator='etd2',
        dt=0.05,
        bounds=bounds,
        grid_shape=grid_shape,
        shear_rate=shear_rate,
    )
    cfg_ref = _write_integrator_config(
        cfg_dir / 'ref',
        spatial_dim=spatial_dim,
        particles_path=particles_path,
        integrator='etd2',
        dt=0.0025,
        bounds=bounds,
        grid_shape=grid_shape,
        shear_rate=shear_rate,
    )

    out_etd = tmp_path / f'out_etd_cmp_{spatial_dim}d'
    out_etd2 = tmp_path / f'out_etd2_cmp_{spatial_dim}d'
    out_ref = tmp_path / f'out_ref_cmp_{spatial_dim}d'
    _run_solver_for_dim(spatial_dim, cfg_etd, out_etd)
    _run_solver_for_dim(spatial_dim, cfg_etd2, out_etd2)
    _run_solver_for_dim(spatial_dim, cfg_ref, out_ref)

    p_etd = _final_particle_position(out_etd, spatial_dim)
    p_etd2 = _final_particle_position(out_etd2, spatial_dim)
    p_ref = _final_particle_position(out_ref, spatial_dim)
    assert float(np.linalg.norm(p_etd2 - p_ref)) <= float(np.linalg.norm(p_etd - p_ref)) + 1e-12

def test_2d_single_wall_reflection_uses_physical_hit_velocity_and_time(tmp_path: Path):
    config_path = _write_single_reflection_case(tmp_path, spatial_dim=2)
    out_dir = tmp_path / 'out_single_reflection_2d'
    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    _assert_single_reflection_result(out_dir)

def test_3d_single_wall_reflection_uses_physical_hit_velocity_and_time(tmp_path: Path):
    config_path = _write_single_reflection_case(tmp_path, spatial_dim=3)
    out_dir = tmp_path / 'out_single_reflection_3d'
    run_solver_3d_from_yaml(config_path, output_dir=out_dir)
    _assert_single_reflection_result(out_dir)

def test_apply_wall_hit_step_subtracts_hit_time_across_multiple_hits():
    context = _wall_hit_context(particle_id=1)

    result1 = _apply_test_wall_hit_step(
        context,
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
        max_wall_hits_per_step=5,
    )
    v1 = result1.velocity
    assert result1.should_break is False
    assert result1.entered_contact is False
    assert result1.remaining_dt == pytest.approx(0.18, abs=1e-15)

    result2 = _apply_test_wall_hit_step(
        context,
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
        max_wall_hits_per_step=5,
    )
    assert result2.should_break is False
    assert result2.entered_contact is False
    assert result2.remaining_dt == pytest.approx(0.16, abs=1e-15)

def test_apply_wall_hit_step_records_minimal_max_hit_diagnostics():
    context = _wall_hit_context(particle_id=42)
    collision_diagnostics = context['collision_diagnostics']
    max_hit_rows = context['max_hit_rows']

    result = _apply_test_wall_hit_step(
        context,
        max_wall_hits_per_step=1,
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
    context = _wall_hit_context(particle_id=42)
    collision_diagnostics = context['collision_diagnostics']

    result = _apply_test_wall_hit_step(
        context,
        v_hit=np.asarray([-2.0, 1.0], dtype=np.float64),
        hit_count=1,
        total_hit_count=1,
        hit_part_ids=[7],
        hit_outcomes=['reflected_specular'],
        max_wall_hits_per_step=2,
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
    context = _wall_hit_context(particle_id=42, spatial_dim=3)
    collision_diagnostics = context['collision_diagnostics']

    result = _apply_test_wall_hit_step(
        context,
        hit=np.asarray([0.0, 0.5, 0.5], dtype=np.float64),
        n_out=np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
        v_hit=np.asarray([-2.0, 1.0, 0.5], dtype=np.float64),
        hit_count=1,
        total_hit_count=1,
        hit_part_ids=[7],
        hit_outcomes=['reflected_specular'],
        max_wall_hits_per_step=2,
    )

    assert result.should_break is True
    assert result.entered_contact is True
    assert result.remaining_dt == pytest.approx(0.0, abs=1e-15)
    assert int(collision_diagnostics['max_hits_reached_count']) == 0
    assert int(collision_diagnostics['contact_sliding_count']) == 1
    assert result.contact_part_id == 7
    assert result.contact_normal == pytest.approx([-1.0, 0.0, 0.0], abs=1e-15)
    assert np.dot(result.velocity, np.asarray([-1.0, 0.0, 0.0], dtype=np.float64)) == pytest.approx(0.0, abs=1e-15)


def _release_grace_square_runtime() -> SimpleNamespace:
    axes = (np.asarray([0.0, 0.5, 1.0], dtype=np.float64),) * 2
    valid_mask = np.ones((3, 3), dtype=bool)
    edges = _square_boundary_edges()
    geometry = GeometryND(
        spatial_dim=2,
        coordinate_system='cartesian_xy',
        axes=axes,
        valid_mask=valid_mask,
        sdf=-np.ones((3, 3), dtype=np.float64),
        normal_components=(np.zeros((3, 3), dtype=np.float64), np.ones((3, 3), dtype=np.float64)),
        nearest_boundary_part_id_map=np.ones((3, 3), dtype=np.int32),
        boundary_edges=edges,
        boundary_edge_part_ids=np.asarray([1, 2, 3, 4], dtype=np.int32),
        boundary_loops_2d=build_boundary_loops_2d(edges),
    )
    return SimpleNamespace(
        geometry_provider=GeometryProviderND(geometry=geometry, kind='test'),
        field_provider=None,
        wall_catalog=None,
        gas=SimpleNamespace(density_kgm3=1.0, dynamic_viscosity_Pas=1.0e-5, temperature=300.0),
    )


def _run_release_grace_collision(
    *,
    x_start: tuple[float, float] = (0.95, 0.5),
    v_start: tuple[float, float] = (1.0, 0.0),
    source_part_id: int = 2,
    release_time_s: float = 0.0,
    grace_time_s: float = 0.06,
    clearance_m: float = 0.06,
) -> tuple[object, dict[str, object], list[dict[str, object]]]:
    runtime = _release_grace_square_runtime()
    service = build_boundary_service(runtime, spatial_dim=2, on_boundary_tol_m=1.0e-9, triangle_surface_3d=None)
    compiled = _compile_runtime_arrays(runtime, spatial_dim=2)
    diagnostics = initial_collision_diagnostics()
    wall_rows: list[dict[str, object]] = []
    x0 = np.asarray(x_start, dtype=np.float64)
    v0 = np.asarray(v_start, dtype=np.float64)
    dt = 0.1
    x1 = x0 + dt * v0
    step = ProcessStepRow(step_id=1, step_name='run', start_s=0.0, end_s=1.0)
    result = _advance_colliding_particle(
        runtime=runtime,
        step=step,
        particles=None,
        particle_index=0,
        rng=np.random.default_rng(123),
        t=dt,
        x_start=x0,
        v_start=v0,
        dt_step=dt,
        spatial_dim=2,
        compiled=compiled,
        integrator_mode=int(get_integrator_spec('drag_relaxation').mode),
        base_adaptive_substep_enabled=0,
        adaptive_substep_tau_ratio=1.0,
        adaptive_substep_max_splits=0,
        min_remaining_dt_ratio=0.0,
        tau_p_i=1.0e9,
        particle_diameter_i=1.0e-6,
        particle_density_i=1000.0,
        particle_mass_i=1.0e-15,
        particle_id_i=1,
        source_part_id_i=int(source_part_id),
        release_time_i=float(release_time_s),
        release_grace=ReleaseGracePlan(
            enabled=True,
            grace_time_s=float(grace_time_s),
            clearance_m=float(clearance_m),
            min_outward_normal_speed_mps=0.0,
        ),
        particle_stick_probability_i=0.0,
        flow_scale_particle_i=1.0,
        drag_scale_particle_i=1.0,
        body_scale_particle_i=1.0,
        global_flow_scale=1.0,
        global_drag_tau_scale=1.0,
        global_body_accel_scale=1.0,
        body_accel=np.zeros(2, dtype=np.float64),
        min_tau_p_s=1.0e-12,
        gas_density_kgm3=1.0,
        gas_mu_pas=1.0e-5,
        gas_temperature_K=300.0,
        gas_molecular_mass_kg=60.0 * 1.66053906660e-27,
        drag_model_mode=int(DRAG_MODEL_STOKES),
        valid_mask_retry_then_stop_enabled=False,
        initial_x_next=x1,
        initial_v_next=v0,
        initial_stage_points=np.asarray([x1], dtype=np.float64),
        initial_valid_mask_status=int(VALID_MASK_STATUS_CLEAN),
        initial_primary_hit=None,
        initial_primary_hit_counted=False,
        inside_fn=service.inside,
        strict_inside_fn=service.inside_strict,
        primary_hit_fn=service.polyline_hit,
        nearest_projection_fn=service.nearest_projection,
        primary_hit_counter_key=service.primary_hit_counter_key,
        collision_diagnostics=diagnostics,
        max_hit_rows=[],
        wall_rows=wall_rows,
        coating_summary_rows=[],
        wall_law_counts={},
        wall_summary_counts={},
        stuck=np.asarray([False], dtype=bool),
        absorbed=np.asarray([False], dtype=bool),
        escaped=np.asarray([False], dtype=bool),
        active=np.asarray([True], dtype=bool),
        max_wall_hits_per_step=5,
        epsilon_offset_m=1.0e-6,
        on_boundary_tol_m=1.0e-9,
        triangle_surface_3d=None,
    )
    return result, diagnostics, wall_rows


def test_release_grace_skips_outward_same_source_inside_grace():
    result, diagnostics, wall_rows = _run_release_grace_collision()

    assert result.total_hits == 0
    assert result.position[0] > 1.0
    assert int(diagnostics['source_surface_release_skip_count']) == 1
    assert int(diagnostics['source_surface_release_skip_blocked_count']) == 0
    assert wall_rows == []


def test_release_grace_blocks_outward_same_source_outside_grace():
    result, diagnostics, wall_rows = _run_release_grace_collision(grace_time_s=0.01)

    assert result.total_hits == 1
    assert int(diagnostics['source_surface_release_skip_count']) == 0
    assert int(diagnostics['source_surface_release_skip_blocked_count']) == 1
    assert diagnostics['source_surface_release_skip_blocked_reasons'] == {'outside_grace_time': 1}
    assert len(wall_rows) == 1


def test_release_grace_handles_inward_same_source_reimpact_as_wall_event():
    result, diagnostics, wall_rows = _run_release_grace_collision(
        x_start=(1.05, 0.5),
        v_start=(-1.0, 0.0),
        grace_time_s=0.2,
    )

    assert result.total_hits >= 1
    assert int(diagnostics['source_surface_release_skip_count']) == 0
    assert int(diagnostics['source_surface_release_skip_blocked_count']) >= 1
    assert diagnostics['source_surface_release_skip_blocked_reasons']['not_outward'] == 1
    assert len(wall_rows) >= 1


def test_release_grace_leaves_unrelated_wall_hit_unchanged():
    result, diagnostics, wall_rows = _run_release_grace_collision(source_part_id=1)

    assert result.total_hits == 1
    assert int(diagnostics['source_surface_release_skip_count']) == 0
    assert int(diagnostics['source_surface_release_skip_blocked_count']) == 0
    assert diagnostics['source_surface_release_skip_blocked_reasons'] == {}
    assert len(wall_rows) == 1


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
    options = _solver_options_for_test()
    state = _runtime_state_for_test(
        x=np.asarray([[1.0e-6, 0.0]], dtype=np.float64),
        v=np.zeros((1, 2), dtype=np.float64),
        contact_sliding=True,
        contact_endpoint_stopped=True,
        contact_edge_index=0,
        contact_part_id=1,
        contact_normal=np.asarray([[-1.0, 0.0]], dtype=np.float64),
    )

    _advance_contact_sliding_particles_2d(
        runtime=runtime,
        state=state,
        options=options,
        compiled=compiled,
        boundary_service=build_boundary_service(runtime, spatial_dim=2, on_boundary_tol_m=1.0e-6, triangle_surface_3d=None),
        tau_p=np.asarray([1.0e-3], dtype=np.float64),
        particle_diameter=np.asarray([1.0e-6], dtype=np.float64),
        particle_density=np.asarray([1200.0], dtype=np.float64),
        particle_mass=np.asarray([1200.0 * np.pi * (1.0e-6) ** 3 / 6.0], dtype=np.float64),
        dep_particle_rel_permittivity=np.asarray([np.nan], dtype=np.float64),
        thermophoretic_coeff=np.asarray([np.nan], dtype=np.float64),
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
            'gas_temperature_K': 300.0,
            'gas_molecular_mass_kg': 60.0 * 1.66053906660e-27,
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

def test_contact_sliding_virtual_mass_uses_effective_mass_2d():
    axes = _regular_axes(2)
    valid_mask = _regular_valid_mask(2)
    quantities = {
        'ux': np.zeros_like(valid_mask, dtype=np.float64),
        'uy': np.zeros_like(valid_mask, dtype=np.float64),
        'rho_g': np.ones_like(valid_mask, dtype=np.float64) * 1000.0,
        'mu': np.ones_like(valid_mask, dtype=np.float64) * 1.0e-3,
        'T': np.ones_like(valid_mask, dtype=np.float64) * 300.0,
    }
    field_provider = _regular_field_provider_from_arrays(axes, valid_mask, quantities)
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
        gas=SimpleNamespace(density_kgm3=1000.0, dynamic_viscosity_Pas=1.0e-3, temperature=300.0),
    )
    force_runtime = ForceRuntimeParameters(virtual_mass_enabled=True, virtual_mass_coefficient=0.5)
    compiled = _compile_runtime_arrays(runtime, spatial_dim=2, force_runtime=force_runtime)
    options = _solver_options_for_test(dt=0.1, t_end=0.1, force_runtime=force_runtime)
    state = _runtime_state_for_test(
        x=np.asarray([[1.0e-6, 0.5]], dtype=np.float64),
        v=np.zeros((1, 2), dtype=np.float64),
        contact_sliding=True,
        contact_edge_index=0,
        contact_part_id=1,
        contact_normal=np.asarray([[-1.0, 0.0]], dtype=np.float64),
    )

    _advance_contact_sliding_particles_2d(
        runtime=runtime,
        state=state,
        options=options,
        compiled=compiled,
        boundary_service=build_boundary_service(runtime, spatial_dim=2, on_boundary_tol_m=1.0e-6, triangle_surface_3d=None),
        tau_p=np.asarray([1.0], dtype=np.float64),
        particle_diameter=np.asarray([1.0e-6], dtype=np.float64),
        particle_density=np.asarray([1000.0], dtype=np.float64),
        particle_mass=np.asarray([1000.0 * np.pi * (1.0e-6) ** 3 / 6.0], dtype=np.float64),
        dep_particle_rel_permittivity=np.asarray([np.nan], dtype=np.float64),
        thermophoretic_coeff=np.asarray([np.nan], dtype=np.float64),
        flow_scale_particle=np.asarray([1.0], dtype=np.float64),
        drag_scale_particle=np.asarray([1.0], dtype=np.float64),
        body_scale_particle=np.asarray([1.0], dtype=np.float64),
        phys={
            'flow_scale': 1.0,
            'drag_tau_scale': 1.0,
            'body_accel_scale': 1.0,
            'min_tau_p_s': 1.0e-6,
            'gas_density_kgm3': 1000.0,
            'gas_mu_pas': 1.0e-3,
            'gas_temperature_K': 300.0,
            'gas_molecular_mass_kg': 60.0 * 1.66053906660e-27,
        },
        body_accel=np.asarray([0.0, 10.0], dtype=np.float64),
        dt_step=0.1,
        t_next=0.1,
    )

    mass_factor = 1.0 + 0.5 * 1000.0 / 1000.0
    expected_tangent_velocity = 10.0 * (1.0 - np.exp(-0.1 / mass_factor))
    assert state.v[0, 0] == pytest.approx(0.0, abs=1.0e-12)
    assert state.v[0, 1] == pytest.approx(expected_tangent_velocity, rel=1.0e-6)
    assert bool(state.contact_sliding[0]) is True

def test_3d_contact_sliding_advances_on_triangle_face():
    axes = tuple(np.asarray([0.0, 0.5, 1.0], dtype=np.float64) for _ in range(3))
    valid_mask = np.ones((3, 3, 3), dtype=bool)
    quantities = {
        'ux': np.zeros_like(valid_mask, dtype=np.float64),
        'uy': np.zeros_like(valid_mask, dtype=np.float64),
        'uz': np.zeros_like(valid_mask, dtype=np.float64),
        'E_x': np.zeros_like(valid_mask, dtype=np.float64),
        'E_y': np.ones_like(valid_mask, dtype=np.float64) * 2.0,
        'E_z': np.zeros_like(valid_mask, dtype=np.float64),
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
    options = _solver_options_for_test(dt=0.1, t_end=0.1)
    state = _runtime_state_for_test(
        x=np.asarray([[1.0e-6, 0.5, 0.5]], dtype=np.float64),
        v=np.asarray([[0.0, 0.1, 0.0]], dtype=np.float64),
        contact_sliding=True,
        contact_part_id=1,
        contact_normal=np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float64),
    )

    _advance_contact_sliding_particles_3d(
        runtime=runtime,
        state=state,
        options=options,
        compiled=compiled,
        boundary_service=boundary_service,
        tau_p=np.asarray([1.0], dtype=np.float64),
        particle_diameter=np.asarray([1.0e-6], dtype=np.float64),
        particle_density=np.asarray([1200.0], dtype=np.float64),
        particle_mass=np.asarray([1200.0 * np.pi * (1.0e-6) ** 3 / 6.0], dtype=np.float64),
        dep_particle_rel_permittivity=np.asarray([np.nan], dtype=np.float64),
        thermophoretic_coeff=np.asarray([np.nan], dtype=np.float64),
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
            'gas_temperature_K': 300.0,
            'gas_molecular_mass_kg': 60.0 * 1.66053906660e-27,
        },
        body_accel=np.zeros(3, dtype=np.float64),
        dt_step=0.1,
        t_next=0.1,
        electric_q_over_m_particle=np.asarray([1.0], dtype=np.float64),
    )

    assert bool(state.contact_sliding[0]) is True
    assert int(state.contact_edge_index[0]) >= 0
    assert state.x[0, 0] == pytest.approx(1.0e-6, abs=1.0e-12)
    assert state.x[0, 1] > 0.5
    assert state.v[0, 1] > 0.1
    assert int(state.collision_diagnostics['contact_tangent_step_count']) == 1
    assert int(state.collision_diagnostics['contact_valid_mask_reject_count']) == 0

def test_final_snapshot_matches_t_end_when_not_divisible(tmp_path: Path):
    out_dir = tmp_path / 'out_2d'
    config_path = _write_minimal_2d_config(
        tmp_path,
        solver_updates={'dt': 0.3, 't_end': 1.0, 'save_every': 1, 'plot_particle_limit': 3},
        output_updates={'mode': 'debug'},
    )
    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    save_frames = _read_table(out_dir / 'save_frames.csv')
    assert save_frames['time_s'].iloc[-1] == pytest.approx(1.0, abs=1e-12)

def test_plot_particle_limit_zero_skips_trajectory_plot(tmp_path: Path):
    out_dir = tmp_path / 'out_2d_no_plot'
    config_path = _write_minimal_2d_config(
        tmp_path,
        solver_updates={'plot_particle_limit': 0},
        output_updates={'mode': 'debug'},
    )
    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    assert not (out_dir / 'trajectories.png').exists()

def test_3d_wall_events_use_boundary_part_ids(tmp_path: Path):
    out_dir = tmp_path / 'out_3d'
    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml',
        mutate=lambda cfg: cfg.setdefault('output', {}).update({'mode': 'debug'}),
    )
    run_solver_3d_from_yaml(config_path, output_dir=out_dir)
    wall_events = _read_table(out_dir / 'wall_events.csv')
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
    coating_summary = _read_table(out_dir / 'coating_summary_by_part.csv')
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
    config_path = _write_minimal_2d_config(
        tmp_path,
        output_updates={'artifact_mode': 'minimal'},
    )

    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    report = _solver_report(out_dir)

    assert (out_dir / 'final_particles.csv').exists()
    assert (out_dir / 'wall_summary.json').exists()
    assert (out_dir / 'coating_summary_by_part.csv').exists()
    coating_summary = _read_table(out_dir / 'coating_summary_by_part.csv')
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


def test_default_standard_output_suppresses_deep_artifacts(tmp_path: Path):
    out_dir = tmp_path / 'out_2d_standard_default'
    config_path = _write_minimal_2d_config(tmp_path / 'cfg_standard_default')

    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    report = _solver_report(out_dir)

    assert report['output_mode'] == 'standard'
    assert int(report['output_debug_enabled']) == 0
    assert int(report['output_minimal_enabled']) == 0
    assert (out_dir / 'final_particles.csv').exists()
    assert (out_dir / 'solver_report.json').exists()
    assert (out_dir / 'prepared_runtime_summary.json').exists()
    assert (out_dir / 'wall_summary.json').exists()
    assert (out_dir / 'coating_summary_by_part.csv').exists()
    assert not (out_dir / 'positions_2d.npy').exists()
    assert not (out_dir / 'save_frames.csv').exists()
    assert not (out_dir / 'wall_events.csv').exists()
    assert not (out_dir / 'runtime_step_summary.csv').exists()
    assert not (out_dir / 'source_particle_diagnostics.csv').exists()
    assert not (out_dir / 'collision_diagnostics.json').exists()
    assert not (out_dir / 'force_contributions.csv').exists()


def test_artifact_mode_full_keeps_debug_artifacts(tmp_path: Path):
    out_dir = tmp_path / 'out_2d_full_alias'
    config_path = _write_minimal_2d_config(
        tmp_path / 'cfg_full_alias',
        output_updates={'artifact_mode': 'full'},
    )

    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    report = _solver_report(out_dir)

    assert report['output_mode'] == 'debug'
    assert int(report['output_debug_enabled']) == 1
    assert (out_dir / 'positions_2d.npy').exists()
    assert (out_dir / 'save_frames.csv').exists()
    assert (out_dir / 'wall_events.csv').exists()
    assert (out_dir / 'runtime_step_summary.csv').exists()
    assert (out_dir / 'source_particle_diagnostics.csv').exists()
    assert (out_dir / 'collision_diagnostics.json').exists()
    assert (out_dir / 'force_contributions.csv').exists()


def test_release_grace_minimal_output_keeps_wall_trace_suppressed(tmp_path: Path):
    out_dir = tmp_path / 'out_release_grace_minimal'
    config_path = _write_minimal_2d_config(
        tmp_path / 'cfg_release_grace_minimal',
        solver_updates={
            'release_grace': {
                'enabled': True,
                'grace_time_s': 1.0e-4,
            },
        },
        output_updates={'artifact_mode': 'minimal'},
    )

    run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    report = _solver_report(out_dir)

    assert int(report['source_surface_release_grace_enabled']) == 1
    assert not (out_dir / 'wall_events.csv').exists()
    assert not (out_dir / 'source_surface_release_events.csv').exists()


def test_solver_reports_schiller_naumann_drag_model(tmp_path: Path):
    out_dir = tmp_path / 'out_2d_schiller_naumann'
    config_path = _write_minimal_2d_config(
        tmp_path,
        solver_updates={'drag_model': 'schiller_naumann', 't_end': 0.05, 'save_every': 1, 'plot_particle_limit': 0},
        output_updates={'write_collision_diagnostics': 1},
    )

    report, _prepared = run_solver_2d_from_yaml(config_path, output_dir=out_dir)
    solver_report = _solver_report(out_dir)
    diag = _collision_diagnostics(out_dir)

    assert report['drag_model'] == 'schiller_naumann'
    assert solver_report['drag_model'] == 'schiller_naumann'
    assert diag['drag_model'] == 'schiller_naumann'

def test_segment_output_filenames_are_sanitized(tmp_path: Path):
    steps_path = _write_rows_csv(
        tmp_path / 'process_steps_sanitized.csv',
        [
            {
                'step_id': 1,
                'step_name': 'etch',
                'start_s': 0.0,
                'end_s': 1.0,
                'output_segment_name': 'etch:phase/1',
            }
        ],
    )
    out_dir = tmp_path / 'out_2d_segment_names'
    config_path = _write_minimal_2d_config(
        tmp_path,
        path_updates={'process_steps_csv': str(steps_path.resolve())},
        output_updates={'mode': 'debug'},
    )

    run_solver_2d_from_yaml(config_path, output_dir=out_dir)

    assert (out_dir / 'segments' / 'positions_etch_phase_1_2d.npy').exists()

def test_single_run_segment_does_not_duplicate_positions_output(tmp_path: Path):
    out_dir = tmp_path / 'out_single_segment_no_duplicate'
    config_path = _write_minimal_2d_config(
        tmp_path,
        solver_updates={'t_end': 0.02, 'save_every': 1, 'plot_particle_limit': 0},
        output_updates={'mode': 'debug'},
    )

    run_solver_2d_from_yaml(config_path, output_dir=out_dir)

    assert (out_dir / 'positions_2d.npy').exists()
    assert (out_dir / 'save_frames.csv').exists()
    assert (out_dir / 'segment_summary.csv').exists()
    assert not (out_dir / 'segments' / 'positions_run_2d.npy').exists()

def test_3d_collision_diagnostics_and_max_hits_limit_are_applied(tmp_path: Path):
    particles_path = _write_particle_row(
        tmp_path / 'fast_particles_3d.csv',
        _one_particle_row(spatial_dim=3, vx=80.0, diameter=1e-6, density=1200.0),
    )

    def update_config(cfg):
        cfg.setdefault('paths', {}).update({'particles_csv': str(particles_path.resolve())})
        cfg.setdefault('solver', {}).update(
            {
                'dt': 0.2,
                't_end': 0.2,
                'save_every': 1,
                'min_tau_p_s': 1.0,
                'max_wall_hits_per_step': 1,
                'min_remaining_dt_ratio': 0.0,
            }
        )
        cfg.setdefault('output', {}).update({'write_collision_diagnostics': 1})
        cfg.setdefault('output', {}).update({'write_max_hit_events': 1})

    config_path = _write_config(
        tmp_path,
        ROOT / 'examples' / 'minimal_3d' / 'run_config.yaml',
        mutate=update_config,
    )
    out_dir = tmp_path / 'out_diag_hits_3d'
    run_solver_3d_from_yaml(config_path, output_dir=out_dir)
    diag = _collision_diagnostics(out_dir)
    assert int(diag['max_wall_hits_per_step']) == 1
    assert int(diag['triangle_hit_count']) >= 1
    assert int(diag['max_hits_reached_count']) >= 1
    assert int(diag['max_hit_event_summary']['event_count']) == int(diag['max_hits_reached_count'])
    assert int(diag['max_hit_event_summary']['unique_particle_count']) >= 1
    max_hit_events = _read_table(out_dir / 'max_hit_events.csv')
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
    final_df = _final_particles(out_dir)
    pts = final_df.loc[final_df['escaped'] == 0, ['x', 'y', 'z']].to_numpy(dtype=np.float64)
    inside = [point_inside_surface(surface, p, on_boundary_tol=2.0e-6)[0] for p in pts]
    assert bool(np.all(np.asarray(inside, dtype=bool)))
