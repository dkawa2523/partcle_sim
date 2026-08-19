from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import particle_tracer_unified.solvers._force_field_sources as force_field_sources
from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    QuantitySeriesND,
    RegularFieldND,
    TriangleMeshField2D,
)
from particle_tracer_unified.core.field_backend import field_backend_report
from particle_tracer_unified.core.triangle_mesh_sampling_2d import (
    build_triangle_candidate_grid,
)
from particle_tracer_unified.domain import StageFields
from particle_tracer_unified.force_models import parse_native_force_model
from particle_tracer_unified.solvers.compiled_backend_types import (
    RegularRectilinearCompiledBackend,
    TriangleMesh2DCompiledBackend,
)
from particle_tracer_unified.solvers.field_compilation import compile_runtime_backend
from particle_tracer_unified.solvers.force_field_assembly import (
    sample_compiled_acceleration_vector,
    sample_compiled_acceleration_vectors,
    sample_compiled_stage_fields,
)
from particle_tracer_unified.solvers.forces import (
    ForceRuntimeParameters,
    resolve_force_catalog,
)


def _catalog_for(provider: FieldProviderND, force_name: str):
    model = parse_native_force_model(
        {"model": "stokes"},
        {force_name: {"enabled": True}},
        spatial_dim=2,
    )
    return resolve_force_catalog(model, field_provider=provider, spatial_dim=2)


_TEST_DIAMETER_M = 1.0e-6
_TEST_GAS_MOLECULAR_MASS_KG = 60.0 * 1.66053906660e-27


def _particle_mass(density_kgm3: float) -> np.ndarray:
    return np.asarray(
        [density_kgm3 * np.pi * _TEST_DIAMETER_M**3 / 6.0], dtype=np.float64
    )


def _series(name: str, values: np.ndarray) -> QuantitySeriesND:
    return QuantitySeriesND(
        name=name,
        unit="",
        times=np.asarray([0.0], dtype=np.float64),
        data=np.asarray(values, dtype=np.float64),
    )


def _accelerating_field_provider() -> FieldProviderND:
    axes = (
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )
    xx, _yy = np.meshgrid(axes[0], axes[1], indexing="ij")
    shape = (1, 3, 3)
    field = RegularFieldND(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        axis_names=("x", "y"),
        axes=axes,
        valid_mask=np.ones((3, 3), dtype=bool),
        quantities={
            "ux": _series("ux", xx.reshape(shape)),
            "uy": _series("uy", np.zeros(shape, dtype=np.float64)),
            "rho_g": _series("rho_g", np.ones(shape, dtype=np.float64) * 2.0),
        },
    )
    return FieldProviderND(field=field)


def _varying_field_provider() -> FieldProviderND:
    axes = (
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
    )
    xx, yy = np.meshgrid(axes[0], axes[1], indexing="ij")
    shape = (1, 3, 3)
    field = RegularFieldND(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        axis_names=("x", "y"),
        axes=axes,
        valid_mask=np.ones((3, 3), dtype=bool),
        quantities={
            "ux": _series("ux", yy.reshape(shape)),
            "uy": _series("uy", np.zeros(shape, dtype=np.float64)),
            "E_x": _series("E_x", xx.reshape(shape)),
            "E_y": _series("E_y", np.zeros(shape, dtype=np.float64)),
            "T": _series("T", (300.0 + 20.0 * xx).reshape(shape)),
        },
    )
    return FieldProviderND(field=field)


def _varying_3d_field_provider() -> FieldProviderND:
    axes = tuple(np.asarray([0.0, 0.5, 1.0], dtype=np.float64) for _ in range(3))
    xx, yy, zz = np.meshgrid(*axes, indexing="ij")
    shape = (1, 3, 3, 3)
    quantities = {
        "ux": yy,
        "uy": zz,
        "uz": xx,
        "E_x": xx,
        "E_y": 2.0 * yy,
        "E_z": 3.0 * zz,
        "rho_g": np.full_like(xx, 2.0),
        "mu": np.full_like(xx, 1.8e-5),
        "T": 300.0 + 10.0 * xx + 20.0 * yy + 30.0 * zz,
    }
    field = RegularFieldND(
        spatial_dim=3,
        coordinate_system="cartesian_xyz",
        axis_names=("x", "y", "z"),
        axes=axes,
        valid_mask=np.ones((3, 3, 3), dtype=bool),
        quantities={
            name: _series(name, values.reshape(shape))
            for name, values in quantities.items()
        },
    )
    return FieldProviderND(field=field)


def _triangle_accelerating_field_provider() -> FieldProviderND:
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
    (
        accel_origin,
        accel_cell_size,
        accel_shape,
        accel_offsets,
        accel_triangle_indices,
    ) = build_triangle_candidate_grid(
        vertices,
        triangles,
    )
    times = np.asarray([0.0], dtype=np.float64)
    field = TriangleMeshField2D(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        quantities={
            "ux": QuantitySeriesND("ux", "m/s", times, vertices[:, 0].reshape(1, -1)),
            "uy": QuantitySeriesND(
                "uy", "m/s", times, np.zeros((1, vertices.shape[0]), dtype=np.float64)
            ),
            "rho_g": QuantitySeriesND(
                "rho_g",
                "kg/m^3",
                times,
                np.ones((1, vertices.shape[0]), dtype=np.float64) * 2.0,
            ),
            "T": QuantitySeriesND(
                "T", "K", times, (300.0 + 20.0 * vertices[:, 0]).reshape(1, -1)
            ),
            "E_x": QuantitySeriesND("E_x", "V/m", times, vertices[:, 0].reshape(1, -1)),
            "E_y": QuantitySeriesND(
                "E_y", "V/m", times, np.zeros((1, vertices.shape[0]), dtype=np.float64)
            ),
        },
        accel_origin=accel_origin,
        accel_cell_size=accel_cell_size,
        accel_shape=accel_shape,
        accel_cell_offsets=accel_offsets,
        accel_triangle_indices=accel_triangle_indices,
        metadata={"field_backend_kind": "triangle_mesh_2d"},
    )
    return FieldProviderND(field=field, kind="triangle_mesh_test")


def _triangle_shear_field_provider() -> FieldProviderND:
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
    (
        accel_origin,
        accel_cell_size,
        accel_shape,
        accel_offsets,
        accel_triangle_indices,
    ) = build_triangle_candidate_grid(
        vertices,
        triangles,
    )
    times = np.asarray([0.0], dtype=np.float64)
    field = TriangleMeshField2D(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        quantities={
            "ux": QuantitySeriesND("ux", "m/s", times, vertices[:, 1].reshape(1, -1)),
            "uy": QuantitySeriesND(
                "uy", "m/s", times, np.zeros((1, vertices.shape[0]), dtype=np.float64)
            ),
            "rho_g": QuantitySeriesND(
                "rho_g",
                "kg/m^3",
                times,
                np.ones((1, vertices.shape[0]), dtype=np.float64),
            ),
            "mu": QuantitySeriesND(
                "mu",
                "Pa*s",
                times,
                np.ones((1, vertices.shape[0]), dtype=np.float64) * 1.8e-5,
            ),
        },
        accel_origin=accel_origin,
        accel_cell_size=accel_cell_size,
        accel_shape=accel_shape,
        accel_cell_offsets=accel_offsets,
        accel_triangle_indices=accel_triangle_indices,
        metadata={"field_backend_kind": "triangle_mesh_2d"},
    )
    return FieldProviderND(field=field, kind="triangle_mesh_test")


def _runtime(provider: FieldProviderND) -> SimpleNamespace:
    field = provider.field
    if isinstance(field, TriangleMeshField2D):
        axes = (
            np.asarray([0.0, 1.0], dtype=np.float64),
            np.asarray([0.0, 1.0], dtype=np.float64),
        )
        valid_mask = np.ones((2, 2), dtype=bool)
    else:
        axes = field.axes
        valid_mask = field.valid_mask
    return SimpleNamespace(
        geometry_provider=SimpleNamespace(
            geometry=SimpleNamespace(axes=axes, valid_mask=valid_mask)
        ),
        field_provider=provider,
        gas=SimpleNamespace(
            density_kgm3=1.0, dynamic_viscosity_Pas=1.8e-5, temperature=300.0
        ),
    )


def test_regular_stage_field_assembly_preserves_numeric_contracts() -> None:
    provider = _varying_3d_field_provider()
    force_runtime = ForceRuntimeParameters(
        pressure_gradient_enabled=True,
        virtual_mass_enabled=True,
        thermophoresis_enabled=True,
        dielectrophoresis_enabled=True,
        lift_enabled=True,
    )
    backend = compile_runtime_backend(
        _runtime(provider),
        3,
        force_runtime=force_runtime,
    )
    points = np.asarray(
        [[0.25, 0.5, 0.75], [0.75, 0.25, 0.5]],
        dtype=np.float64,
    )
    supplied_flow = np.asarray(
        [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0]],
        dtype=np.float32,
    )

    fields = sample_compiled_stage_fields(
        backend,
        3,
        0.0,
        points,
        force_runtime=force_runtime,
        include_electric=True,
        flow_velocity=supplied_flow,
        gas_density_kgm3=1.0,
        gas_mu_pas=1.8e-5,
        gas_temperature_K=300.0,
    )

    expected_shapes = {
        "electric_field": (2, 3),
        "gas_density": (2,),
        "dynamic_viscosity": (2,),
        "temperature": (2,),
        "fluid_acceleration": (2, 3),
        "flow_time_derivative": (2, 3),
        "flow_velocity_gradient": (2, 3, 3),
        "temperature_gradient": (2, 3),
        "electric_magnitude_squared_gradient": (2, 3),
        "flow_velocity": (2, 3),
        "vorticity": (2, 3),
    }
    assert {
        name: value.shape for name, value in fields.values.items()
    } == expected_shapes
    assert all(value.dtype == np.dtype(np.float64) for value in fields.values.values())
    assert all(np.all(np.isfinite(value)) for value in fields.values.values())
    np.testing.assert_array_equal(fields.values["flow_velocity"], supplied_flow)
    np.testing.assert_allclose(
        fields.values["fluid_acceleration"],
        [[0.75, 0.25, 0.5], [0.5, 0.75, 0.25]],
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        fields.values["temperature_gradient"],
        [[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]],
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        fields.values["electric_magnitude_squared_gradient"],
        [[0.5, 4.0, 13.5], [1.5, 2.0, 9.0]],
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        fields.values["vorticity"],
        np.full((2, 3), -1.0),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_triangle_stage_field_assembly_preserves_fallback_calculations() -> None:
    provider = _triangle_accelerating_field_provider()
    force_runtime = ForceRuntimeParameters(
        pressure_gradient_enabled=True,
        virtual_mass_enabled=True,
        thermophoresis_enabled=True,
        dielectrophoresis_enabled=True,
        lift_enabled=True,
    )
    backend = compile_runtime_backend(
        _runtime(provider),
        2,
        force_runtime=force_runtime,
    )
    points = np.asarray([[0.25, 0.25], [0.75, 0.5]], dtype=np.float64)

    fields = sample_compiled_stage_fields(
        backend,
        2,
        0.0,
        points,
        force_runtime=force_runtime,
        include_electric=True,
        gas_density_kgm3=1.0,
        gas_mu_pas=1.8e-5,
        gas_temperature_K=300.0,
    )

    assert all(value.dtype == np.dtype(np.float64) for value in fields.values.values())
    assert all(np.all(np.isfinite(value)) for value in fields.values.values())
    np.testing.assert_allclose(
        fields.values["fluid_acceleration"],
        [[0.25, 0.0], [0.75, 0.0]],
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        fields.values["temperature_gradient"],
        [[20.0, 0.0], [20.0, 0.0]],
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        fields.values["electric_magnitude_squared_gradient"],
        [[0.5, 0.0], [1.5, 0.0]],
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_array_equal(fields.values["vorticity"], np.zeros((2, 3)))


def test_regular_stage_field_diagnostics_keep_force_phase_order() -> None:
    provider = _varying_field_provider()
    force_runtime = ForceRuntimeParameters(
        pressure_gradient_enabled=True,
        virtual_mass_enabled=True,
        thermophoresis_enabled=True,
        dielectrophoresis_enabled=True,
        lift_enabled=True,
    )
    backend = compile_runtime_backend(
        _runtime(provider),
        2,
        force_runtime=force_runtime,
    )
    assert isinstance(backend, RegularRectilinearCompiledBackend)
    missing_pressure_and_later_fields = replace(
        backend,
        fluid_accel_x=None,
        fluid_accel_y=None,
        du_dt_x=None,
        du_dt_y=None,
        grad_T_x=None,
        grad_T_y=None,
        grad_E2_x=None,
        grad_E2_y=None,
        vorticity_z=None,
    )

    with pytest.raises(
        ValueError,
        match="pressure_gradient requires fluid material acceleration",
    ):
        sample_compiled_stage_fields(
            missing_pressure_and_later_fields,
            2,
            0.0,
            np.asarray([[0.5, 0.5]], dtype=np.float64),
            force_runtime=force_runtime,
            gas_density_kgm3=1.0,
            gas_mu_pas=1.8e-5,
            gas_temperature_K=300.0,
        )


def test_pressure_gradient_force_samples_fluid_material_acceleration() -> None:
    provider = _accelerating_field_provider()
    force_runtime = ForceRuntimeParameters(pressure_gradient_enabled=True)
    backend = compile_runtime_backend(
        _runtime(provider), 2, force_runtime=force_runtime
    )
    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        force_runtime=force_runtime,
        particle_diameter=np.asarray([_TEST_DIAMETER_M], dtype=np.float64),
        particle_density=np.asarray([2000.0], dtype=np.float64),
        particle_mass=_particle_mass(2000.0),
    )

    assert accel.shape == (1, 2)
    assert accel[0, 0] == pytest.approx(2.0 / 2000.0 * 0.5)
    assert accel[0, 1] == pytest.approx(0.0)


def test_virtual_mass_force_samples_particle_path_fluid_acceleration() -> None:
    provider = _accelerating_field_provider()
    force_runtime = ForceRuntimeParameters(
        virtual_mass_enabled=True, virtual_mass_coefficient=0.5
    )
    backend = compile_runtime_backend(
        _runtime(provider), 2, force_runtime=force_runtime
    )
    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        force_runtime=force_runtime,
        particle_diameter=np.asarray([_TEST_DIAMETER_M], dtype=np.float64),
        particle_density=np.asarray([2000.0], dtype=np.float64),
        particle_mass=_particle_mass(2000.0),
        velocity=np.asarray([[1.5, 0.0]], dtype=np.float64),
    )

    assert accel.shape == (1, 2)
    assert accel[0, 0] == pytest.approx(0.5 * 2.0 / 2000.0 * 1.5)
    assert accel[0, 1] == pytest.approx(0.0)


def test_triangle_mesh_pressure_gradient_samples_fluid_material_acceleration() -> None:
    provider = _triangle_accelerating_field_provider()
    force_runtime = ForceRuntimeParameters(pressure_gradient_enabled=True)
    backend = compile_runtime_backend(
        _runtime(provider), 2, force_runtime=force_runtime
    )
    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        force_runtime=force_runtime,
        particle_diameter=np.asarray([_TEST_DIAMETER_M], dtype=np.float64),
        particle_density=np.asarray([2000.0], dtype=np.float64),
        particle_mass=_particle_mass(2000.0),
    )

    assert accel.shape == (1, 2)
    assert accel[0, 0] == pytest.approx(2.0 / 2000.0 * 0.5)
    assert accel[0, 1] == pytest.approx(0.0)


def test_triangle_mesh_pressure_gradient_prefers_exported_fluid_acceleration() -> None:
    provider = _triangle_accelerating_field_provider()
    field = provider.field
    quantities = dict(field.quantities)
    quantities.pop("ux", None)
    quantities.pop("uy", None)
    times = np.asarray([0.0], dtype=np.float64)
    quantities["fluid_accel_x"] = QuantitySeriesND(
        "fluid_accel_x",
        "m/s^2",
        times,
        np.ones((1, field.mesh_vertices.shape[0]), dtype=np.float64) * 10.0,
    )
    quantities["fluid_accel_y"] = QuantitySeriesND(
        "fluid_accel_y",
        "m/s^2",
        times,
        np.ones((1, field.mesh_vertices.shape[0]), dtype=np.float64) * -4.0,
    )
    exported = FieldProviderND(
        field=TriangleMeshField2D(
            spatial_dim=field.spatial_dim,
            coordinate_system=field.coordinate_system,
            mesh_vertices=field.mesh_vertices,
            mesh_triangles=field.mesh_triangles,
            quantities=quantities,
            accel_origin=field.accel_origin,
            accel_cell_size=field.accel_cell_size,
            accel_shape=field.accel_shape,
            accel_cell_offsets=field.accel_cell_offsets,
            accel_triangle_indices=field.accel_triangle_indices,
            time_mode=field.time_mode,
            metadata=field.metadata,
        ),
        kind=provider.kind,
    )
    force_runtime = ForceRuntimeParameters(pressure_gradient_enabled=True)
    catalog = _catalog_for(exported, "pressure_gradient")
    backend = compile_runtime_backend(
        _runtime(exported), 2, force_runtime=force_runtime
    )
    assert isinstance(backend, TriangleMesh2DCompiledBackend)
    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        force_runtime=force_runtime,
        particle_diameter=np.asarray([_TEST_DIAMETER_M], dtype=np.float64),
        particle_density=np.asarray([2000.0], dtype=np.float64),
        particle_mass=_particle_mass(2000.0),
    )

    assert catalog.by_name()["pressure_gradient"].required_fields == (
        "fluid_accel_x",
        "fluid_accel_y",
    )
    assert (
        backend.triangle_gradient_sources["fluid_acceleration"] == "exported_quantity"
    )
    assert accel[0, 0] == pytest.approx(2.0 / 2000.0 * 10.0)
    assert accel[0, 1] == pytest.approx(2.0 / 2000.0 * -4.0)


def _triangle_pressure_gradient_requires_velocity_or_exported_acceleration() -> None:
    provider = _triangle_accelerating_field_provider()
    field = provider.field
    quantities = dict(field.quantities)
    quantities.pop("ux", None)
    quantities.pop("uy", None)
    no_motion_source = FieldProviderND(
        field=TriangleMeshField2D(
            spatial_dim=field.spatial_dim,
            coordinate_system=field.coordinate_system,
            mesh_vertices=field.mesh_vertices,
            mesh_triangles=field.mesh_triangles,
            quantities=quantities,
            accel_origin=field.accel_origin,
            accel_cell_size=field.accel_cell_size,
            accel_shape=field.accel_shape,
            accel_cell_offsets=field.accel_cell_offsets,
            accel_triangle_indices=field.accel_triangle_indices,
            time_mode=field.time_mode,
            metadata=field.metadata,
        ),
        kind=provider.kind,
    )

    catalog = _catalog_for(no_motion_source, "pressure_gradient")
    assert catalog.enabled("pressure_gradient")
    assert catalog.by_name()["pressure_gradient"].required_fields == ()

    with pytest.raises(ValueError, match="pressure_gradient"):
        compile_runtime_backend(
            _runtime(no_motion_source),
            2,
            force_runtime=ForceRuntimeParameters(pressure_gradient_enabled=True),
        )


globals()[
    "test_triangle_mesh_pressure_gradient_requires_velocity_or_exported_acceleration"
] = _triangle_pressure_gradient_requires_velocity_or_exported_acceleration


def _triangle_virtual_mass_requires_velocity_with_exported_acceleration() -> None:
    provider = _triangle_accelerating_field_provider()
    field = provider.field
    quantities = dict(field.quantities)
    quantities.pop("ux", None)
    quantities.pop("uy", None)
    times = np.asarray([0.0], dtype=np.float64)
    quantities["fluid_accel_x"] = QuantitySeriesND(
        "fluid_accel_x",
        "m/s^2",
        times,
        np.ones((1, field.mesh_vertices.shape[0]), dtype=np.float64),
    )
    quantities["fluid_accel_y"] = QuantitySeriesND(
        "fluid_accel_y",
        "m/s^2",
        times,
        np.zeros((1, field.mesh_vertices.shape[0]), dtype=np.float64),
    )
    exported = FieldProviderND(
        field=TriangleMeshField2D(
            spatial_dim=field.spatial_dim,
            coordinate_system=field.coordinate_system,
            mesh_vertices=field.mesh_vertices,
            mesh_triangles=field.mesh_triangles,
            quantities=quantities,
            accel_origin=field.accel_origin,
            accel_cell_size=field.accel_cell_size,
            accel_shape=field.accel_shape,
            accel_cell_offsets=field.accel_cell_offsets,
            accel_triangle_indices=field.accel_triangle_indices,
            time_mode=field.time_mode,
            metadata=field.metadata,
        ),
        kind=provider.kind,
    )

    catalog = _catalog_for(exported, "virtual_mass")
    assert catalog.enabled("virtual_mass")
    assert catalog.by_name()["virtual_mass"].required_fields == ()

    with pytest.raises(ValueError, match="virtual_mass"):
        compile_runtime_backend(
            _runtime(exported),
            2,
            force_runtime=ForceRuntimeParameters(virtual_mass_enabled=True),
        )


globals()[
    "test_triangle_mesh_virtual_mass_requires_velocity_even_with_"
    "exported_fluid_acceleration"
] = _triangle_virtual_mass_requires_velocity_with_exported_acceleration


def test_triangle_mesh_gradient_alias_is_reported_as_exported_quantity() -> None:
    provider = _triangle_accelerating_field_provider()
    field = provider.field
    quantities = dict(field.quantities)
    times = np.asarray([0.0], dtype=np.float64)
    quantities["dT_dx"] = QuantitySeriesND(
        "dT_dx",
        "K/m",
        times,
        np.ones((1, field.mesh_vertices.shape[0]), dtype=np.float64) * 20.0,
    )
    quantities["dT_dy"] = QuantitySeriesND(
        "dT_dy",
        "K/m",
        times,
        np.zeros((1, field.mesh_vertices.shape[0]), dtype=np.float64),
    )
    alias_provider = FieldProviderND(
        field=TriangleMeshField2D(
            spatial_dim=field.spatial_dim,
            coordinate_system=field.coordinate_system,
            mesh_vertices=field.mesh_vertices,
            mesh_triangles=field.mesh_triangles,
            quantities=quantities,
            accel_origin=field.accel_origin,
            accel_cell_size=field.accel_cell_size,
            accel_shape=field.accel_shape,
            accel_cell_offsets=field.accel_cell_offsets,
            accel_triangle_indices=field.accel_triangle_indices,
            time_mode=field.time_mode,
            metadata=field.metadata,
        ),
        kind=provider.kind,
    )

    report = field_backend_report(alias_provider)
    backend = compile_runtime_backend(_runtime(alias_provider), 2)

    assert report["triangle_gradient_sources"]["grad_T"] == "exported_quantity"
    assert backend.triangle_gradient_sources["grad_T"] == "exported_quantity"
    assert backend.triangle_gradient_sources == report["triangle_gradient_sources"]


def test_triangle_mesh_virtual_mass_samples_particle_path_fluid_acceleration() -> None:
    provider = _triangle_accelerating_field_provider()
    force_runtime = ForceRuntimeParameters(
        virtual_mass_enabled=True, virtual_mass_coefficient=0.5
    )
    backend = compile_runtime_backend(
        _runtime(provider), 2, force_runtime=force_runtime
    )
    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        force_runtime=force_runtime,
        particle_diameter=np.asarray([_TEST_DIAMETER_M], dtype=np.float64),
        particle_density=np.asarray([2000.0], dtype=np.float64),
        particle_mass=_particle_mass(2000.0),
        velocity=np.asarray([[1.5, 0.0]], dtype=np.float64),
    )

    assert accel.shape == (1, 2)
    assert accel[0, 0] == pytest.approx(0.5 * 2.0 / 2000.0 * 1.5)
    assert accel[0, 1] == pytest.approx(0.0)


def test_triangle_mesh_electric_force_samples_field() -> None:
    provider = _triangle_accelerating_field_provider()
    backend = compile_runtime_backend(_runtime(provider), 2)
    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        electric_q_over_m=np.asarray([3.0], dtype=np.float64),
        particle_mass=np.asarray([1.0], dtype=np.float64),
    )

    assert accel.shape == (1, 2)
    assert accel[0, 0] == pytest.approx(1.5)
    assert accel[0, 1] == pytest.approx(0.0)


def test_triangle_mesh_thermophoresis_samples_temperature_gradient() -> None:
    provider = _triangle_accelerating_field_provider()
    force_runtime = ForceRuntimeParameters(thermophoresis_enabled=True)
    backend = compile_runtime_backend(
        _runtime(provider), 2, force_runtime=force_runtime
    )
    diameter = np.asarray([1.0e-6], dtype=np.float64)
    density = np.asarray([2200.0], dtype=np.float64)
    mass = density * np.pi * diameter**3 / 6.0
    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        force_runtime=force_runtime,
        particle_diameter=diameter,
        particle_density=density,
        particle_mass=mass,
        gas_mu_pas=1.8e-5,
        gas_molecular_mass_kg=_TEST_GAS_MOLECULAR_MASS_KG,
    )

    assert np.all(np.isfinite(accel))
    assert accel[0, 0] < 0.0
    assert accel[0, 1] == pytest.approx(0.0)


def test_triangle_mesh_dielectrophoresis_samples_electric_field_gradient() -> None:
    provider = _triangle_accelerating_field_provider()
    force_runtime = ForceRuntimeParameters(
        dielectrophoresis_enabled=True,
        dep_particle_rel_permittivity=3.9,
        dep_medium_rel_permittivity=1.0,
    )
    backend = compile_runtime_backend(
        _runtime(provider), 2, force_runtime=force_runtime
    )
    diameter = np.asarray([1.0e-6], dtype=np.float64)
    density = np.asarray([2200.0], dtype=np.float64)
    mass = density * np.pi * diameter**3 / 6.0
    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        force_runtime=force_runtime,
        particle_diameter=diameter,
        particle_density=density,
        particle_mass=mass,
        dep_particle_rel_permittivity=np.asarray([3.9], dtype=np.float64),
    )

    assert np.all(np.isfinite(accel))
    assert accel[0, 0] > 0.0
    assert accel[0, 1] == pytest.approx(0.0)


def test_dielectrophoresis_rejects_missing_particle_permittivity() -> None:
    provider = _varying_field_provider()
    force_runtime = ForceRuntimeParameters(
        dielectrophoresis_enabled=True,
        dep_medium_rel_permittivity=1.0,
    )
    backend = compile_runtime_backend(
        _runtime(provider), 2, force_runtime=force_runtime
    )

    with pytest.raises(
        ValueError, match="explicit positive particle relative permittivity"
    ):
        sample_compiled_acceleration_vectors(
            backend,
            2,
            0.0,
            np.asarray([[0.5, 0.5]], dtype=np.float64),
            force_runtime=force_runtime,
            particle_diameter=np.asarray([1.0e-6], dtype=np.float64),
            particle_mass=np.asarray([1.0e-15], dtype=np.float64),
        )


def test_triangle_mesh_lift_samples_velocity_gradient() -> None:
    provider = _triangle_shear_field_provider()
    force_runtime = ForceRuntimeParameters(lift_enabled=True)
    backend = compile_runtime_backend(
        _runtime(provider), 2, force_runtime=force_runtime
    )
    diameter = np.asarray([1.0e-6], dtype=np.float64)
    density = np.asarray([2200.0], dtype=np.float64)
    mass = density * np.pi * diameter**3 / 6.0
    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        force_runtime=force_runtime,
        particle_diameter=diameter,
        particle_density=density,
        particle_mass=mass,
        velocity=np.asarray([[1.5, 0.0]], dtype=np.float64),
    )

    assert np.all(np.isfinite(accel))
    assert accel[0, 0] == pytest.approx(0.0)
    assert accel[0, 1] < 0.0


def test_comsol_style_forces_add_expected_acceleration_directions() -> None:
    provider = _varying_field_provider()
    force_runtime = ForceRuntimeParameters(
        thermophoresis_enabled=True,
        dielectrophoresis_enabled=True,
        lift_enabled=True,
        dep_particle_rel_permittivity=3.9,
        dep_medium_rel_permittivity=1.0,
    )
    backend = compile_runtime_backend(
        _runtime(provider), 2, force_runtime=force_runtime
    )
    positions = np.asarray([[0.5, 0.5]], dtype=np.float64)
    velocity = np.asarray([[1.5, 0.0]], dtype=np.float64)
    diameter = np.asarray([1.0e-6], dtype=np.float64)
    density = np.asarray([2200.0], dtype=np.float64)
    mass = density * np.pi * diameter**3 / 6.0

    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        positions,
        force_runtime=force_runtime,
        particle_diameter=diameter,
        particle_density=density,
        particle_mass=mass,
        dep_particle_rel_permittivity=np.asarray([3.9], dtype=np.float64),
        thermophoretic_coeff=np.asarray([1.0], dtype=np.float64),
        velocity=velocity,
        gas_density_kgm3=1.0,
        gas_mu_pas=1.8e-5,
        gas_temperature_K=300.0,
        gas_molecular_mass_kg=_TEST_GAS_MOLECULAR_MASS_KG,
    )

    assert accel.shape == (1, 2)
    assert accel[0, 0] < 0.0
    assert accel[0, 1] < 0.0


@pytest.mark.parametrize(
    "force_runtime",
    [
        ForceRuntimeParameters(thermophoresis_enabled=True),
        ForceRuntimeParameters(
            dielectrophoresis_enabled=True,
            dep_particle_rel_permittivity=3.9,
            dep_medium_rel_permittivity=1.0,
        ),
        ForceRuntimeParameters(lift_enabled=True),
    ],
    ids=("thermophoresis", "dielectrophoresis", "saffman_lift"),
)
def test_non_drag_forces_use_mass_density_equivalent_sphere_diameter(
    force_runtime: ForceRuntimeParameters,
) -> None:
    provider = _varying_field_provider()
    backend = compile_runtime_backend(
        _runtime(provider), 2, force_runtime=force_runtime
    )
    physical_diameter = 1.0e-6
    particle_density = np.asarray([2_200.0], dtype=np.float64)
    particle_mass = particle_density * np.pi * physical_diameter**3 / 6.0

    def _sample(drag_diameter: float) -> np.ndarray:
        return sample_compiled_acceleration_vectors(
            backend,
            2,
            0.0,
            np.asarray([[0.5, 0.5]], dtype=np.float64),
            force_runtime=force_runtime,
            particle_diameter=np.asarray([drag_diameter], dtype=np.float64),
            particle_density=particle_density,
            particle_mass=particle_mass,
            dep_particle_rel_permittivity=np.asarray([3.9], dtype=np.float64),
            thermophoretic_coeff=np.asarray([1.0], dtype=np.float64),
            velocity=np.asarray([[1.5, 0.0]], dtype=np.float64),
            gas_density_kgm3=1.0,
            gas_mu_pas=1.8e-5,
            gas_temperature_K=300.0,
            gas_molecular_mass_kg=_TEST_GAS_MOLECULAR_MASS_KG,
        )

    sphere_result = _sample(physical_diameter)
    aerodynamic_result = _sample(4.0 * physical_diameter)

    assert np.linalg.norm(sphere_result) > 0.0
    np.testing.assert_allclose(
        aerodynamic_result,
        sphere_result,
        rtol=2.0e-14,
        atol=0.0,
    )


def test_scalar_regular_grid_extra_forces_match_batch_pipeline() -> None:
    provider = _varying_field_provider()
    force_runtime = ForceRuntimeParameters(
        thermophoresis_enabled=True,
        dielectrophoresis_enabled=True,
        lift_enabled=True,
        dep_particle_rel_permittivity=3.9,
        dep_medium_rel_permittivity=1.0,
    )
    backend = compile_runtime_backend(
        _runtime(provider), 2, force_runtime=force_runtime
    )
    position = np.asarray([0.5, 0.5], dtype=np.float64)
    velocity = np.asarray([1.5, 0.0], dtype=np.float64)
    diameter = 1.0e-6
    density = 2200.0
    mass = density * np.pi * diameter**3 / 6.0

    scalar = sample_compiled_acceleration_vector(
        backend,
        2,
        0.0,
        position,
        force_runtime=force_runtime,
        particle_diameter=diameter,
        particle_density=density,
        particle_mass=mass,
        dep_particle_rel_permittivity=3.9,
        thermophoretic_coeff=1.0,
        velocity=velocity,
        gas_density_kgm3=1.0,
        gas_mu_pas=1.8e-5,
        gas_temperature_K=300.0,
        gas_molecular_mass_kg=4.65e-26,
    )
    batch = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        position.reshape(1, 2),
        force_runtime=force_runtime,
        particle_diameter=np.asarray([diameter], dtype=np.float64),
        particle_density=np.asarray([density], dtype=np.float64),
        particle_mass=np.asarray([mass], dtype=np.float64),
        dep_particle_rel_permittivity=np.asarray([3.9], dtype=np.float64),
        thermophoretic_coeff=np.asarray([1.0], dtype=np.float64),
        velocity=velocity.reshape(1, 2),
        gas_density_kgm3=1.0,
        gas_mu_pas=1.8e-5,
        gas_temperature_K=300.0,
        gas_molecular_mass_kg=4.65e-26,
    )

    assert scalar.tolist() == pytest.approx(batch[0].tolist())


def test_force_pipeline_reuses_pre_sampled_stage_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _varying_field_provider()
    force_runtime = ForceRuntimeParameters(thermophoresis_enabled=True)
    backend = compile_runtime_backend(
        _runtime(provider), 2, force_runtime=force_runtime
    )
    points = np.asarray([[0.5, 0.5]], dtype=np.float64)
    stage_fields = StageFields(
        points_m=points,
        time_s=0.0,
        values={
            "gas_density": np.asarray([1.2]),
            "dynamic_viscosity": np.asarray([1.8e-5]),
            "temperature": np.asarray([300.0]),
        },
        supported=np.asarray([True]),
    )

    def fail_duplicate_gas_sample(*_args, **_kwargs):
        raise AssertionError("gas fields were sampled twice for one force stage")

    monkeypatch.setattr(
        force_field_sources,
        "sample_compiled_gas_properties_vectors",
        fail_duplicate_gas_sample,
    )
    acceleration = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        points,
        force_runtime=force_runtime,
        particle_diameter=np.asarray([1.0e-6]),
        particle_mass=np.asarray([1.0e-15]),
        gas_molecular_mass_kg=_TEST_GAS_MOLECULAR_MASS_KG,
        stage_fields=stage_fields,
    )

    assert np.all(np.isfinite(acceleration))


def test_3d_lift_uses_vector_vorticity_cross_product() -> None:
    provider = _varying_3d_field_provider()
    params = ForceRuntimeParameters(lift_enabled=True)
    backend = compile_runtime_backend(_runtime(provider), 3, force_runtime=params)
    acceleration = sample_compiled_acceleration_vectors(
        backend,
        3,
        0.0,
        np.asarray([[0.25, 0.5, 0.75]], dtype=np.float64),
        force_runtime=params,
        particle_diameter=np.asarray([1.0e-6]),
        particle_mass=np.asarray([1.2e-15]),
        velocity=np.asarray([[1.0, -0.5, 0.25]], dtype=np.float64),
        gas_density_kgm3=2.0,
        gas_mu_pas=1.8e-5,
    )

    assert acceleration[0, 0] < 0.0
    assert acceleration[0, 1] < 0.0
    assert acceleration[0, 2] > 0.0


def test_3d_optional_forces_use_same_pipeline_for_scalar_and_batch() -> None:
    provider = _varying_3d_field_provider()
    params = ForceRuntimeParameters(
        pressure_gradient_enabled=True,
        virtual_mass_enabled=True,
        thermophoresis_enabled=True,
        dielectrophoresis_enabled=True,
        lift_enabled=True,
        dep_particle_rel_permittivity=3.9,
        dep_medium_rel_permittivity=1.0,
    )
    backend = compile_runtime_backend(_runtime(provider), 3, force_runtime=params)
    points = np.asarray([[0.25, 0.5, 0.75], [0.75, 0.25, 0.5]], dtype=np.float64)
    velocity = np.asarray([[1.0, -0.5, 0.25], [-0.25, 0.75, 1.0]], dtype=np.float64)
    diameter = np.asarray([1.0e-6, 1.5e-6], dtype=np.float64)
    mass = np.asarray([1.2e-15, 3.4e-15], dtype=np.float64)
    density = np.asarray([1800.0, 2400.0], dtype=np.float64)
    common = {
        "force_runtime": params,
        "gas_density_kgm3": 2.0,
        "gas_mu_pas": 1.8e-5,
        "gas_temperature_K": 300.0,
        "gas_molecular_mass_kg": _TEST_GAS_MOLECULAR_MASS_KG,
    }
    batch = sample_compiled_acceleration_vectors(
        backend,
        3,
        0.0,
        points,
        particle_diameter=diameter,
        particle_density=density,
        particle_mass=mass,
        dep_particle_rel_permittivity=np.asarray([3.9, 3.9]),
        thermophoretic_coeff=np.ones(2),
        velocity=velocity,
        **common,
    )
    scalar = np.asarray(
        [
            sample_compiled_acceleration_vector(
                backend,
                3,
                0.0,
                points[idx],
                particle_diameter=float(diameter[idx]),
                particle_density=float(density[idx]),
                particle_mass=float(mass[idx]),
                dep_particle_rel_permittivity=3.9,
                thermophoretic_coeff=1.0,
                velocity=velocity[idx],
                **common,
            )
            for idx in range(2)
        ]
    )

    assert np.all(np.isfinite(batch))
    assert np.any(np.abs(batch[:, 2]) > 0.0)
    np.testing.assert_allclose(scalar, batch, rtol=1.0e-12, atol=0.0)
