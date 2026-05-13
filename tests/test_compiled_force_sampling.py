from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from particle_tracer_unified.core.datamodel import FieldProviderND, QuantitySeriesND, RegularFieldND, TriangleMeshField2D
from particle_tracer_unified.core.field_backend import field_backend_report
from particle_tracer_unified.core.triangle_mesh_sampling_2d import build_triangle_candidate_grid
from particle_tracer_unified.solvers.compiled_field_backend import (
    compile_runtime_backend,
    sample_compiled_acceleration_vectors,
)
from particle_tracer_unified.solvers.forces import (
    ForceRuntimeParameters,
    build_force_catalog,
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
    accel_origin, accel_cell_size, accel_shape, accel_offsets, accel_triangle_indices = build_triangle_candidate_grid(
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
            "uy": QuantitySeriesND("uy", "m/s", times, np.zeros((1, vertices.shape[0]), dtype=np.float64)),
            "rho_g": QuantitySeriesND("rho_g", "kg/m^3", times, np.ones((1, vertices.shape[0]), dtype=np.float64) * 2.0),
            "T": QuantitySeriesND("T", "K", times, (300.0 + 20.0 * vertices[:, 0]).reshape(1, -1)),
            "E_x": QuantitySeriesND("E_x", "V/m", times, vertices[:, 0].reshape(1, -1)),
            "E_y": QuantitySeriesND("E_y", "V/m", times, np.zeros((1, vertices.shape[0]), dtype=np.float64)),
        },
        accel_origin=accel_origin,
        accel_cell_size=accel_cell_size,
        accel_shape=accel_shape,
        accel_cell_offsets=accel_offsets,
        accel_triangle_indices=accel_triangle_indices,
        metadata={"field_backend_kind": "triangle_mesh_2d", "support_tolerance_m": 2.0e-6},
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
    accel_origin, accel_cell_size, accel_shape, accel_offsets, accel_triangle_indices = build_triangle_candidate_grid(
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
            "uy": QuantitySeriesND("uy", "m/s", times, np.zeros((1, vertices.shape[0]), dtype=np.float64)),
            "rho_g": QuantitySeriesND("rho_g", "kg/m^3", times, np.ones((1, vertices.shape[0]), dtype=np.float64)),
            "mu": QuantitySeriesND("mu", "Pa*s", times, np.ones((1, vertices.shape[0]), dtype=np.float64) * 1.8e-5),
        },
        accel_origin=accel_origin,
        accel_cell_size=accel_cell_size,
        accel_shape=accel_shape,
        accel_cell_offsets=accel_offsets,
        accel_triangle_indices=accel_triangle_indices,
        metadata={"field_backend_kind": "triangle_mesh_2d", "support_tolerance_m": 2.0e-6},
    )
    return FieldProviderND(field=field, kind="triangle_mesh_test")


def _runtime(provider: FieldProviderND) -> SimpleNamespace:
    field = provider.field
    if isinstance(field, TriangleMeshField2D):
        axes = (np.asarray([0.0, 1.0], dtype=np.float64), np.asarray([0.0, 1.0], dtype=np.float64))
        valid_mask = np.ones((2, 2), dtype=bool)
    else:
        axes = field.axes
        valid_mask = field.valid_mask
    return SimpleNamespace(
        geometry_provider=SimpleNamespace(
            geometry=SimpleNamespace(axes=axes, valid_mask=valid_mask)
        ),
        field_provider=provider,
        gas=SimpleNamespace(density_kgm3=1.0, dynamic_viscosity_Pas=1.8e-5, temperature=300.0),
    )


def test_pressure_gradient_force_samples_fluid_material_acceleration() -> None:
    provider = _accelerating_field_provider()
    force_runtime = ForceRuntimeParameters(pressure_gradient_enabled=True)
    backend = compile_runtime_backend(_runtime(provider), 2, force_runtime=force_runtime)
    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        force_runtime=force_runtime,
        particle_density=np.asarray([2000.0], dtype=np.float64),
    )

    assert accel.shape == (1, 2)
    assert accel[0, 0] == pytest.approx(2.0 / 2000.0 * 0.5)
    assert accel[0, 1] == pytest.approx(0.0)


def test_virtual_mass_force_samples_particle_path_fluid_acceleration() -> None:
    provider = _accelerating_field_provider()
    force_runtime = ForceRuntimeParameters(virtual_mass_enabled=True, virtual_mass_coefficient=0.5)
    backend = compile_runtime_backend(_runtime(provider), 2, force_runtime=force_runtime)
    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        force_runtime=force_runtime,
        particle_density=np.asarray([2000.0], dtype=np.float64),
        velocity=np.asarray([[1.5, 0.0]], dtype=np.float64),
    )

    assert accel.shape == (1, 2)
    assert accel[0, 0] == pytest.approx(0.5 * 2.0 / 2000.0 * 1.5)
    assert accel[0, 1] == pytest.approx(0.0)


def test_triangle_mesh_pressure_gradient_samples_fluid_material_acceleration() -> None:
    provider = _triangle_accelerating_field_provider()
    force_runtime = ForceRuntimeParameters(pressure_gradient_enabled=True)
    backend = compile_runtime_backend(_runtime(provider), 2, force_runtime=force_runtime)
    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        force_runtime=force_runtime,
        particle_density=np.asarray([2000.0], dtype=np.float64),
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
    catalog = build_force_catalog(
        {"solver": {"forces": {"pressure_gradient": {"enabled": True}}}},
        field_provider=exported,
        spatial_dim=2,
    )
    backend = compile_runtime_backend(_runtime(exported), 2, force_runtime=force_runtime)
    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        force_runtime=force_runtime,
        particle_density=np.asarray([2000.0], dtype=np.float64),
    )

    assert catalog.by_name()["pressure_gradient"].required_fields == ("fluid_accel_x", "fluid_accel_y")
    assert backend.triangle_gradient_sources["fluid_acceleration"] == "exported_quantity"
    assert accel[0, 0] == pytest.approx(2.0 / 2000.0 * 10.0)
    assert accel[0, 1] == pytest.approx(2.0 / 2000.0 * -4.0)


def test_triangle_mesh_pressure_gradient_requires_velocity_or_exported_acceleration() -> None:
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

    with pytest.raises(ValueError, match="pressure_gradient"):
        build_force_catalog(
            {"solver": {"forces": {"pressure_gradient": {"enabled": True}}}},
            field_provider=no_motion_source,
            spatial_dim=2,
        )

    with pytest.raises(ValueError, match="pressure_gradient"):
        compile_runtime_backend(
            _runtime(no_motion_source),
            2,
            force_runtime=ForceRuntimeParameters(pressure_gradient_enabled=True),
        )


def test_triangle_mesh_virtual_mass_requires_velocity_even_with_exported_fluid_acceleration() -> None:
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

    with pytest.raises(ValueError, match="virtual_mass"):
        build_force_catalog(
            {"solver": {"forces": {"virtual_mass": {"enabled": True}}}},
            field_provider=exported,
            spatial_dim=2,
        )

    with pytest.raises(ValueError, match="virtual_mass"):
        compile_runtime_backend(
            _runtime(exported),
            2,
            force_runtime=ForceRuntimeParameters(virtual_mass_enabled=True),
        )


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
    force_runtime = ForceRuntimeParameters(virtual_mass_enabled=True, virtual_mass_coefficient=0.5)
    backend = compile_runtime_backend(_runtime(provider), 2, force_runtime=force_runtime)
    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        force_runtime=force_runtime,
        particle_density=np.asarray([2000.0], dtype=np.float64),
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
    )

    assert accel.shape == (1, 2)
    assert accel[0, 0] == pytest.approx(1.5)
    assert accel[0, 1] == pytest.approx(0.0)


def test_triangle_mesh_thermophoresis_samples_temperature_gradient() -> None:
    provider = _triangle_accelerating_field_provider()
    force_runtime = ForceRuntimeParameters(thermophoresis_enabled=True)
    backend = compile_runtime_backend(_runtime(provider), 2, force_runtime=force_runtime)
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
    backend = compile_runtime_backend(_runtime(provider), 2, force_runtime=force_runtime)
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


def test_triangle_mesh_lift_samples_velocity_gradient() -> None:
    provider = _triangle_shear_field_provider()
    force_runtime = ForceRuntimeParameters(lift_enabled=True)
    backend = compile_runtime_backend(_runtime(provider), 2, force_runtime=force_runtime)
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
    assert accel[0, 1] > 0.0


def test_comsol_style_forces_add_expected_acceleration_directions() -> None:
    provider = _varying_field_provider()
    force_runtime = ForceRuntimeParameters(
        thermophoresis_enabled=True,
        dielectrophoresis_enabled=True,
        lift_enabled=True,
        dep_particle_rel_permittivity=3.9,
        dep_medium_rel_permittivity=1.0,
    )
    backend = compile_runtime_backend(_runtime(provider), 2, force_runtime=force_runtime)
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
    )

    assert accel.shape == (1, 2)
    assert accel[0, 0] < 0.0
    assert accel[0, 1] > 0.0
