from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    QuantitySeriesND,
    TriangleMeshField2D,
)
from particle_tracer_unified.core.field_sampling import VALID_MASK_STATUS_HARD_INVALID
from particle_tracer_unified.core.triangle_mesh_sampling_2d import (
    build_triangle_candidate_grid,
)
from particle_tracer_unified.solvers.base_field_sampling import (
    sample_compiled_flow_vectors,
    sample_compiled_valid_mask_statuses,
)
from particle_tracer_unified.solvers.compiled_backend_types import (
    RegularRectilinearCompiledBackend,
)
from particle_tracer_unified.solvers.field_compilation import compile_runtime_backend
from particle_tracer_unified.solvers.force_field_assembly import (
    sample_compiled_acceleration_vectors,
)
from particle_tracer_unified.solvers.forces import ForceRuntimeParameters


def _triangle_runtime(
    *,
    space_scale: float = 1.0,
    time_origin: float = 0.0,
    time_step: float = 1.0,
    vertices: np.ndarray | None = None,
    quantities: dict[str, QuantitySeriesND] | None = None,
):
    scale = float(space_scale)
    mesh_vertices = np.asarray(
        vertices
        if vertices is not None
        else scale
        * np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        ),
        dtype=np.float64,
    )
    mesh_triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    origin, cell_size, shape, offsets, indices = build_triangle_candidate_grid(
        mesh_vertices,
        mesh_triangles,
    )
    times = np.asarray([time_origin, time_origin + time_step], dtype=np.float64)
    normalized_x = mesh_vertices[:, 0] / scale
    default_quantities = {
        "ux": QuantitySeriesND(
            "ux",
            "m/s",
            times,
            np.stack((normalized_x, normalized_x + 1.0), axis=0),
        ),
        "uy": QuantitySeriesND(
            "uy",
            "m/s",
            times,
            np.zeros((2, mesh_vertices.shape[0]), dtype=np.float64),
        ),
        "rho_g": QuantitySeriesND(
            "rho_g",
            "kg/m^3",
            np.asarray([0.0]),
            np.ones((1, mesh_vertices.shape[0]), dtype=np.float64),
        ),
    }
    field = TriangleMeshField2D(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        mesh_vertices=mesh_vertices,
        mesh_triangles=mesh_triangles,
        quantities=default_quantities if quantities is None else quantities,
        accel_origin=origin,
        accel_cell_size=cell_size,
        accel_shape=shape,
        accel_cell_offsets=offsets,
        accel_triangle_indices=indices,
        time_mode="transient",
        metadata={"field_backend_kind": "triangle_mesh_2d"},
    )
    axes = (
        np.asarray([np.min(mesh_vertices[:, 0]), np.max(mesh_vertices[:, 0])]),
        np.asarray([np.min(mesh_vertices[:, 1]), np.max(mesh_vertices[:, 1])]),
    )
    runtime = SimpleNamespace(
        geometry_provider=SimpleNamespace(
            geometry=SimpleNamespace(axes=axes, valid_mask=np.ones((2, 2), dtype=bool))
        ),
        field_provider=FieldProviderND(field=field, kind="triangle_mesh_test"),
        gas=SimpleNamespace(
            density_kgm3=float("nan"),
            dynamic_viscosity_Pas=float("nan"),
            temperature=float("nan"),
        ),
    )
    return runtime, field


@pytest.mark.parametrize("scale", [1.0e-13, 1.0, 1.0e3])
def test_triangle_p1_space_and_time_derivatives_are_similarity_scale_invariant(
    scale: float,
) -> None:
    runtime, _field = _triangle_runtime(space_scale=scale, time_step=scale)
    params = ForceRuntimeParameters(pressure_gradient_enabled=True)
    backend = compile_runtime_backend(runtime, 2, force_runtime=params)

    acceleration = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.5 * scale,
        np.asarray([[0.25 * scale, 0.25 * scale]], dtype=np.float64),
        force_runtime=params,
        particle_diameter=np.asarray([1.0e-6]),
        particle_density=np.asarray([2000.0]),
        particle_mass=np.asarray([1.0e-15]),
    )

    # ux=x/scale+t/scale gives du/dt=1/scale and ux*du/dx=0.75/scale.
    expected_x = (1.0 / 2000.0) * (1.75 / scale)
    np.testing.assert_allclose(
        acceleration,
        np.asarray([[expected_x, 0.0]]),
        rtol=2.0e-12,
        atol=0.0,
    )


def test_triangle_gradient_rejects_relative_degeneracy_with_triangle_context() -> None:
    vertices = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0e-16], [0.0, 1.0]],
        dtype=np.float64,
    )
    runtime, _field = _triangle_runtime(vertices=vertices)

    with pytest.raises(
        ValueError,
        match=r"semantic quantity 'fluid_acceleration'.*triangle 0.*Gram determinant",
    ):
        compile_runtime_backend(
            runtime,
            2,
            force_runtime=ForceRuntimeParameters(pressure_gradient_enabled=True),
        )


def test_triangle_transient_rejects_unresolved_time_ulp_with_row_and_triangle() -> None:
    time_origin = 1.0e16
    time_step = float(np.nextafter(time_origin, np.inf) - time_origin)
    runtime, _field = _triangle_runtime(
        time_origin=time_origin,
        time_step=time_step,
    )
    backend = compile_runtime_backend(runtime, 2)

    with pytest.raises(
        ValueError,
        match=(
            r"semantic quantity 'flow_velocity\.x'.*row 0, triangle [01]"
            r".*time interval"
        ),
    ):
        sample_compiled_flow_vectors(
            backend,
            2,
            time_origin,
            np.asarray([[0.25, 0.25]], dtype=np.float64),
        )


def test_triangle_declared_nonfinite_flow_is_not_repaired_to_zero() -> None:
    runtime, field = _triangle_runtime()
    quantities = dict(field.quantities)
    bad_ux = np.asarray(quantities["ux"].data, dtype=np.float64).copy()
    bad_ux[0, 0] = np.nan
    quantities["ux"] = QuantitySeriesND("ux", "m/s", quantities["ux"].times, bad_ux)
    runtime, _field = _triangle_runtime(quantities=quantities)
    backend = compile_runtime_backend(runtime, 2)

    with pytest.raises(
        ValueError,
        match=r"semantic quantity 'flow_velocity\.x'.*row 0, triangle [01].*non-finite",
    ):
        sample_compiled_flow_vectors(
            backend,
            2,
            0.0,
            np.asarray([[0.25, 0.25]], dtype=np.float64),
        )


def test_triangle_exported_nonfinite_derived_quantity_has_semantic_context() -> None:
    runtime, field = _triangle_runtime()
    vertex_count = field.mesh_vertices.shape[0]
    quantities = {
        "rho_g": field.quantities["rho_g"],
        "fluid_accel_x": QuantitySeriesND(
            "fluid_accel_x",
            "m/s^2",
            np.asarray([0.0]),
            np.full((1, vertex_count), np.inf),
        ),
        "fluid_accel_y": QuantitySeriesND(
            "fluid_accel_y",
            "m/s^2",
            np.asarray([0.0]),
            np.zeros((1, vertex_count)),
        ),
    }
    runtime, _field = _triangle_runtime(quantities=quantities)
    params = ForceRuntimeParameters(pressure_gradient_enabled=True)
    backend = compile_runtime_backend(runtime, 2, force_runtime=params)

    with pytest.raises(
        ValueError,
        match=(
            r"semantic quantity 'fluid_acceleration\.x'.*row 0, triangle [01]"
            r".*non-finite"
        ),
    ):
        sample_compiled_acceleration_vectors(
            backend,
            2,
            0.0,
            np.asarray([[0.25, 0.25]], dtype=np.float64),
            force_runtime=params,
            particle_diameter=np.asarray([1.0e-6]),
            particle_density=np.asarray([2000.0]),
            particle_mass=np.asarray([1.0e-15]),
        )


def test_triangle_absent_flow_is_zero_and_outside_support_is_flagged() -> None:
    """An absent quantity is zero; an outside point is unsupported.

    Outside the mesh the value query clamps to the nearest element instead of
    returning ``NaN``: a trial step that crosses a wall necessarily lands
    outside, and the wall hit that replaces it can only be localized from a
    finite trajectory.  The guard against using that value as physics is the
    support status, which stays ``HARD_INVALID``.
    """

    runtime, field = _triangle_runtime()
    runtime_without_flow, _field = _triangle_runtime(
        quantities={"rho_g": field.quantities["rho_g"]}
    )
    absent_backend = compile_runtime_backend(runtime_without_flow, 2)
    point = np.asarray([[0.25, 0.25]], dtype=np.float64)
    np.testing.assert_array_equal(
        sample_compiled_flow_vectors(absent_backend, 2, 0.0, point),
        np.zeros((1, 2)),
    )

    declared_backend = compile_runtime_backend(runtime, 2)
    outside = np.asarray([[1.25, 0.25]], dtype=np.float64)
    sampled = sample_compiled_flow_vectors(declared_backend, 2, 0.0, outside)
    inside_values = sample_compiled_flow_vectors(declared_backend, 2, 0.0, point)
    assert np.all(np.isfinite(sampled))
    assert not np.allclose(sampled, np.zeros_like(sampled)) or np.allclose(
        inside_values, np.zeros_like(inside_values)
    )
    assert sample_compiled_valid_mask_statuses(declared_backend, outside).tolist() == [
        int(VALID_MASK_STATUS_HARD_INVALID)
    ]


def test_regular_declared_nonfinite_flow_uses_same_clean_support_contract() -> None:
    valid = np.ones((2, 2), dtype=bool)
    ux = np.zeros((1, 2, 2), dtype=np.float64)
    ux[0, 0, 0] = np.nan
    backend = RegularRectilinearCompiledBackend(
        axes=(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])),
        times=np.asarray([0.0]),
        ux=ux,
        uy=np.zeros_like(ux),
        gas_density=np.ones_like(ux),
        gas_mu=np.ones_like(ux),
        gas_temperature=np.ones_like(ux),
        valid_mask=valid,
        core_valid_mask=valid,
    )

    with pytest.raises(
        ValueError,
        match=r"regular field semantic quantity 'flow_velocity'.*rows \[0\]",
    ):
        sample_compiled_flow_vectors(
            backend,
            2,
            0.0,
            np.asarray([[0.25, 0.25]], dtype=np.float64),
        )
