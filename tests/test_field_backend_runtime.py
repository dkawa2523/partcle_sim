from __future__ import annotations

import regression_test as _regression_helpers

globals().update({
    name: value
    for name, value in vars(_regression_helpers).items()
    if not name.startswith("__")
})

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
    point_near_outside = np.asarray([1.03, 0.25], dtype=np.float64)
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
        metadata={'field_backend_kind': 'triangle_mesh_2d', 'support_tolerance_m': 0.05},
    )
    value = sample_triangle_mesh_series(field.quantities['ux'], field, point_inside, 0.5, mode='linear')
    assert float(value) == pytest.approx(1.75, abs=1e-12)
    assert int(sample_triangle_mesh_status(field, point_inside)) == int(VALID_MASK_STATUS_CLEAN)
    assert int(sample_triangle_mesh_status(field, point_near_outside)) == int(VALID_MASK_STATUS_CLEAN)
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
    axes = _regular_axes(2)
    valid_mask = _regular_valid_mask(2)
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
    runtime = SimpleNamespace(
        geometry_provider=geometry_provider,
        field_provider=field_provider,
        config_payload={'solver': {'field_backend_mode': 'regular_grid'}},
    )
    compiled = _compile_runtime_arrays(runtime, spatial_dim=2)

    assert isinstance(compiled, RegularRectilinearCompiledBackend)
    assert compiled.backend_kind == 'regular_rectilinear'
    assert compiled.valid_mask.shape == (3, 3)
    assert compiled.core_valid_mask.shape == (3, 3)

def test_compile_runtime_arrays_rejects_regular_grid_when_triangle_mode_requested():
    axes = _regular_axes(2)
    valid_mask = _regular_valid_mask(2)
    field_provider = _regular_field_provider_from_arrays(
        axes,
        valid_mask,
        quantities={
            'ux': np.zeros((3, 3), dtype=np.float64),
            'uy': np.zeros((3, 3), dtype=np.float64),
        },
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
        config_payload={'solver': {'field_backend_mode': 'triangle_mesh'}},
    )

    with pytest.raises(ValueError, match='field_backend_mode=triangle_mesh'):
        _compile_runtime_arrays(runtime, spatial_dim=2)

def test_compile_runtime_arrays_samples_electric_force_from_particle_q_over_m_2d():
    axes = _regular_axes(2)
    valid_mask = _regular_valid_mask(2)
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
    runtime = SimpleNamespace(
        geometry_provider=geometry_provider,
        field_provider=field_provider,
        config_payload={'solver': {'field_backend_mode': 'regular_grid'}},
    )

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
    axes = _regular_axes(2)
    valid_mask = _regular_valid_mask(2)
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
    field_provider = _field_provider_with_mismatched_velocity_time_axes()

    report = field_backend_report(field_provider)

    assert report['time_axis']['time_mode'] == 'transient'
    assert report['time_axis']['time_count'] == 2
    assert report['time_axis']['quantity_time_axis_reference'] == 'ux'
    assert report['time_axis']['quantity_time_axis_mismatch_count'] == 1
    assert report['time_axis']['quantity_time_axis_mismatches'] == ['uy']

def test_compile_runtime_arrays_rejects_solver_quantity_time_axis_mismatch():
    axes = _regular_axes(2)
    valid_mask = _regular_valid_mask(2)
    field_provider = _field_provider_with_mismatched_velocity_time_axes()
    geometry_provider = _geometry_provider_from_arrays(
        axes,
        valid_mask,
        sdf=-np.ones((3, 3), dtype=np.float64),
        normal_components=(np.zeros((3, 3), dtype=np.float64), np.ones((3, 3), dtype=np.float64)),
    )
    runtime = SimpleNamespace(
        geometry_provider=geometry_provider,
        field_provider=field_provider,
        config_payload={'solver': {'field_backend_mode': 'regular_grid'}},
    )

    with pytest.raises(ValueError, match='must share one time axis'):
        _compile_runtime_arrays(runtime, spatial_dim=2)

def test_synthetic_transient_field_requires_clean_time_axis():
    axes = _regular_axes(2)

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
    axes = _regular_axes(2)
    valid_mask = _regular_valid_mask(2)
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
    runtime = SimpleNamespace(
        geometry_provider=geometry_provider,
        field_provider=field_provider,
        config_payload={'solver': {'field_backend_mode': 'regular_grid'}},
    )
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

def test_schema_allows_strict_clean_valid_mask_policy():
    schema = json.loads((ROOT / 'schemas' / 'run_config.schema.json').read_text(encoding='utf-8'))
    enum_values = schema['properties']['solver']['properties']['valid_mask_policy']['enum']
    input_contract_schema = schema['properties']['input_contract']['properties']
    force_schema = schema['properties']['solver']['properties']['forces']['properties']

    assert 'strict_clean' in enum_values
    assert input_contract_schema['particle_defaults_policy']['enum'] == ['allow', 'warn', 'error']
    assert force_schema['thermophoresis']['oneOf'][1]['properties']['model']['enum'] == ['talbot', 'continuum']
    assert force_schema['dielectrophoresis']['oneOf'][1]['properties']['model']['enum'] == [
        'dc',
        'ac_clausius_mossotti',
    ]
    assert force_schema['pressure_gradient']['oneOf'][1]['properties']['model']['enum'] == [
        'fluid_material_acceleration',
    ]
    assert force_schema['virtual_mass']['oneOf'][1]['properties']['model']['enum'] == [
        'particle_material_acceleration',
    ]

def test_nonstokes_adaptive_substep_uses_start_tau_eff():
    assert _regular_grid_adaptive_substep_count_for_drag_model(DRAG_MODEL_SCHILLER_NAUMANN) == 16

def test_stokes_adaptive_substep_keeps_tau_stokes():
    assert _regular_grid_adaptive_substep_count_for_drag_model(DRAG_MODEL_STOKES) == 4

def test_regular_3d_kernel_uses_local_gas_density_for_buoyancy():
    axes = _regular_axes(3)
    valid_mask = _regular_valid_mask(3)
    shape = (1, 3, 3, 3)
    field_provider = _regular_field_provider_from_arrays(
        axes,
        valid_mask,
        {
            'ux': np.zeros(shape, dtype=np.float64),
            'uy': np.zeros(shape, dtype=np.float64),
            'uz': np.zeros(shape, dtype=np.float64),
            'rho_g': np.ones(shape, dtype=np.float64) * 1.5,
            'mu': np.ones(shape, dtype=np.float64) * 1.8e-5,
            'T': np.ones(shape, dtype=np.float64) * 300.0,
        },
    )
    runtime = SimpleNamespace(
        geometry_provider=_geometry_provider_from_arrays(
            axes,
            valid_mask,
            sdf=-np.ones(valid_mask.shape, dtype=np.float64),
            normal_components=(
                np.ones(valid_mask.shape, dtype=np.float64),
                np.zeros(valid_mask.shape, dtype=np.float64),
                np.zeros(valid_mask.shape, dtype=np.float64),
            ),
        ),
        field_provider=field_provider,
        gas=SimpleNamespace(density_kgm3=0.1, dynamic_viscosity_Pas=1.8e-5, temperature=300.0),
        config_payload={'solver': {'field_backend_mode': 'regular_grid'}},
    )
    compiled = _compile_runtime_arrays(runtime, spatial_dim=3)
    x = np.asarray([[0.5, 0.5, 0.5]], dtype=np.float64)
    v = np.zeros((1, 3), dtype=np.float64)
    x_trial = np.zeros_like(x)
    v_trial = np.zeros_like(v)
    x_mid_trial = np.zeros_like(x)
    substeps = np.zeros(1, dtype=np.int32)
    mask_status = np.zeros(1, dtype=np.uint8)
    density = np.asarray([3.0], dtype=np.float64)
    diameter = np.asarray([1.0e-6], dtype=np.float64)
    mass = density * np.pi * diameter**3 / 6.0

    _advance_trial_particles(
        spatial_dim=3,
        compiled=compiled,
        x=x,
        v=v,
        active=np.asarray([True], dtype=bool),
        tau_p=np.asarray([1.0e6], dtype=np.float64),
        particle_diameter=diameter,
        particle_mass=mass,
        particle_density=density,
        dep_particle_rel_permittivity=np.asarray([np.nan], dtype=np.float64),
        thermophoretic_coeff=np.asarray([np.nan], dtype=np.float64),
        flow_scale_particle=np.asarray([1.0], dtype=np.float64),
        drag_scale_particle=np.asarray([1.0], dtype=np.float64),
        body_scale_particle=np.asarray([1.0], dtype=np.float64),
        t=1.0e-4,
        dt_step=1.0e-4,
        phys={
            'flow_scale': 0.0,
            'drag_tau_scale': 1.0,
            'body_accel_scale': 1.0,
            'min_tau_p_s': 1.0e-12,
            'gas_density_kgm3': 0.1,
            'gas_mu_pas': 1.8e-5,
            'gas_temperature_K': 300.0,
            'gas_molecular_mass_kg': 60.0 * 1.66053906660e-27,
        },
        body_accel=np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        gas_density_kgm3=0.1,
        gas_mu_pas=1.8e-5,
        drag_model_mode=int(DRAG_MODEL_STOKES),
        integrator_mode=int(get_integrator_spec('etd').mode),
        adaptive_substep_enabled=0,
        adaptive_substep_tau_ratio=0.25,
        adaptive_substep_max_splits=0,
        x_trial=x_trial,
        v_trial=v_trial,
        x_mid_trial=x_mid_trial,
        substep_counts=substeps,
        valid_mask_status_flags=mask_status,
        force_runtime=ForceRuntimeParameters(gravity_buoyancy_enabled=True),
    )

    assert v_trial[0, 0] == pytest.approx(0.5e-4, rel=1.0e-6)
    assert v_trial[0, 1] == pytest.approx(0.0)
    assert v_trial[0, 2] == pytest.approx(0.0)

def test_compile_runtime_arrays_returns_triangle_mesh_backend(tmp_path: Path):
    mesh_path = _write_triangle_mesh_field_npz(tmp_path / 'field_mesh.npz')
    field_provider = build_precomputed_triangle_mesh_field(
        {'npz_path': str(mesh_path)},
        spatial_dim=2,
        coordinate_system='cartesian_xy',
        gas_density_kgm3=1.2,
    )
    axes = _regular_axes(2)
    valid_mask = _regular_valid_mask(2)
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
    assert compiled.gas_mu_source == 'field:mu'
    assert compiled.gas_mu.shape == compiled.ux.shape
    assert sample_compiled_acceleration_vector(compiled, 2, 0.5, np.asarray([0.25, 0.25], dtype=np.float64)).tolist() == pytest.approx([0.0, 0.0])
    assert int(sample_compiled_valid_mask_status(compiled, np.asarray([1.0 + 1.0e-6, 0.25], dtype=np.float64))) == int(VALID_MASK_STATUS_CLEAN)
    assert int(sample_compiled_valid_mask_status(compiled, np.asarray([1.0 + 1.0e-4, 0.25], dtype=np.float64))) == int(VALID_MASK_STATUS_HARD_INVALID)

def test_compile_runtime_arrays_rejects_triangle_mesh_when_regular_grid_mode_requested(tmp_path: Path):
    mesh_path = _write_triangle_mesh_field_npz(tmp_path / 'field_mesh.npz')
    field_provider = build_precomputed_triangle_mesh_field(
        {'npz_path': str(mesh_path)},
        spatial_dim=2,
        coordinate_system='cartesian_xy',
        gas_density_kgm3=1.2,
    )
    axes = _regular_axes(2)
    valid_mask = _regular_valid_mask(2)
    geometry_provider = _geometry_provider_from_arrays(
        axes,
        valid_mask,
        sdf=-np.ones((3, 3), dtype=np.float64),
        normal_components=(np.zeros((3, 3), dtype=np.float64), np.ones((3, 3), dtype=np.float64)),
    )
    runtime = SimpleNamespace(
        geometry_provider=geometry_provider,
        field_provider=field_provider,
        config_payload={'solver': {'field_backend_mode': 'regular_grid'}},
    )

    with pytest.raises(ValueError, match='field_backend_mode=regular_grid'):
        _compile_runtime_arrays(runtime, spatial_dim=2)

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

def test_trial_particle_electric_force_is_not_body_scaled_2d():
    spatial_dim = 2
    axes = tuple(np.asarray([0.0, 0.5, 1.0], dtype=np.float64) for _ in range(spatial_dim))
    valid_mask = np.ones((3, 3), dtype=bool)
    quantities = {
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
    compiled = _compile_runtime_arrays(SimpleNamespace(geometry_provider=geometry_provider, field_provider=field_provider), spatial_dim=2)

    def run_with_body_scale(scale: float) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray([[0.5, 0.5]], dtype=np.float64)
        v = np.zeros((1, 2), dtype=np.float64)
        x_trial = np.zeros_like(x)
        v_trial = np.zeros_like(v)
        _advance_trial_particles(
            spatial_dim=2,
            compiled=compiled,
            x=x,
            v=v,
            active=np.asarray([True], dtype=bool),
            tau_p=np.asarray([1.0], dtype=np.float64),
            particle_diameter=np.asarray([1.0e-6], dtype=np.float64),
            flow_scale_particle=np.asarray([1.0], dtype=np.float64),
            drag_scale_particle=np.asarray([1.0], dtype=np.float64),
            body_scale_particle=np.asarray([scale], dtype=np.float64),
            t=0.1,
            dt_step=0.1,
            phys={'flow_scale': 1.0, 'drag_tau_scale': 1.0, 'body_accel_scale': 4.0, 'min_tau_p_s': 1.0},
            body_accel=np.zeros(2, dtype=np.float64),
            gas_density_kgm3=1.0,
            gas_mu_pas=1.8e-5,
            drag_model_mode=DRAG_MODEL_STOKES,
            integrator_mode=get_integrator_spec('etd2').mode,
            adaptive_substep_enabled=0,
            adaptive_substep_tau_ratio=0.5,
            adaptive_substep_max_splits=4,
            x_trial=x_trial,
            v_trial=v_trial,
            x_mid_trial=np.zeros_like(x),
            substep_counts=np.ones(1, dtype=np.int32),
            valid_mask_status_flags=np.zeros(1, dtype=np.uint8),
            electric_q_over_m_particle=np.asarray([1.0], dtype=np.float64),
        )
        return x_trial, v_trial

    x_base, v_base = run_with_body_scale(1.0)
    x_scaled, v_scaled = run_with_body_scale(3.0)

    assert x_scaled == pytest.approx(x_base)
    assert v_scaled == pytest.approx(v_base)

def test_regular_grid_pressure_gradient_is_evaluated_with_substep_state():
    axes = (np.asarray([0.0, 0.5, 1.0], dtype=np.float64), np.asarray([0.0, 0.5, 1.0], dtype=np.float64))
    xx, _yy = np.meshgrid(axes[0], axes[1], indexing='ij')
    valid_mask = np.ones((3, 3), dtype=bool)
    velocity_slope = 20.0
    quantities = {
        'ux': velocity_slope * xx,
        'uy': np.zeros_like(xx, dtype=np.float64),
        'rho_g': np.ones_like(xx, dtype=np.float64),
    }
    field_provider = _regular_field_provider_from_arrays(axes, valid_mask, quantities=quantities)
    geometry_provider = _geometry_provider_from_arrays(
        axes,
        valid_mask,
        sdf=-np.ones_like(valid_mask, dtype=np.float64),
        normal_components=(np.zeros_like(valid_mask, dtype=np.float64), np.ones_like(valid_mask, dtype=np.float64)),
    )
    force_runtime = ForceRuntimeParameters(pressure_gradient_enabled=True)
    compiled = _compile_runtime_arrays(
        SimpleNamespace(geometry_provider=geometry_provider, field_provider=field_provider),
        spatial_dim=2,
        force_runtime=force_runtime,
    )
    x = np.asarray([[0.25, 0.5]], dtype=np.float64)
    v = np.zeros((1, 2), dtype=np.float64)
    x_trial = np.zeros_like(x)
    v_trial = np.zeros_like(v)
    dt = 0.02
    _advance_trial_particles(
        spatial_dim=2,
        compiled=compiled,
        x=x,
        v=v,
        active=np.asarray([True], dtype=bool),
        tau_p=np.asarray([1.0e12], dtype=np.float64),
        particle_diameter=np.asarray([1.0e-6], dtype=np.float64),
        particle_density=np.asarray([1.0], dtype=np.float64),
        flow_scale_particle=np.asarray([1.0], dtype=np.float64),
        drag_scale_particle=np.asarray([1.0], dtype=np.float64),
        body_scale_particle=np.asarray([1.0], dtype=np.float64),
        t=dt,
        dt_step=dt,
        phys={'flow_scale': 1.0, 'drag_tau_scale': 1.0, 'body_accel_scale': 1.0, 'min_tau_p_s': 1.0e-12},
        body_accel=np.zeros(2, dtype=np.float64),
        gas_density_kgm3=1.0,
        gas_mu_pas=1.8e-5,
        drag_model_mode=DRAG_MODEL_STOKES,
        integrator_mode=get_integrator_spec('etd2').mode,
        adaptive_substep_enabled=0,
        adaptive_substep_tau_ratio=0.5,
        adaptive_substep_max_splits=4,
        x_trial=x_trial,
        v_trial=v_trial,
        x_mid_trial=np.zeros_like(x),
        substep_counts=np.ones(1, dtype=np.int32),
        valid_mask_status_flags=np.zeros(1, dtype=np.uint8),
        force_runtime=force_runtime,
    )

    frozen_accel_x = velocity_slope * velocity_slope * float(x[0, 0])
    frozen_x = float(x[0, 0]) + 0.5 * frozen_accel_x * dt * dt
    assert x_trial[0, 0] > frozen_x
