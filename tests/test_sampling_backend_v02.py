from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from particle_tracer_unified.core.datamodel import QuantitySeriesND, TriangleMeshField2D
from particle_tracer_unified.core.triangle_mesh_sampling_2d import (
    build_triangle_candidate_grid,
)
from particle_tracer_unified.domain import FieldRequest, sample_one
from particle_tracer_unified.providers.precomputed import build_precomputed_field
from particle_tracer_unified.solvers.base_field_sampling import (
    sample_compiled_electric_vectors,
    sample_compiled_flow_vector,
    sample_compiled_flow_vectors,
    sample_compiled_gas_properties_vectors,
)
from particle_tracer_unified.solvers.compiled_backend_types import (
    RegularRectilinearCompiledBackend,
    TriangleMesh2DCompiledBackend,
)
from particle_tracer_unified.solvers.field_compilation import compile_runtime_backend
from particle_tracer_unified.solvers.field_runtime import (
    sample_fields_for_stage,
    sample_scalar_fields_for_stage,
)
from particle_tracer_unified.solvers.runtime_plan import StageFieldPlan
from particle_tracer_unified.solvers.sampling_backend import (
    DYNAMIC_VISCOSITY,
    ELECTRIC_FIELD,
    FLOW_VELOCITY,
    GAS_DENSITY,
    TEMPERATURE,
    VALID_MASK_STATUS,
    CompiledSamplingBackend,
)

ALL_QUANTITIES = FieldRequest(
    (
        FLOW_VELOCITY,
        ELECTRIC_FIELD,
        GAS_DENSITY,
        DYNAMIC_VISCOSITY,
        TEMPERATURE,
        VALID_MASK_STATUS,
    )
)


def _regular_backend(dim: int) -> RegularRectilinearCompiledBackend:
    axes = tuple(np.asarray([0.0, 1.0], dtype=np.float64) for _ in range(dim))
    times = np.asarray([0.0, 1.0], dtype=np.float64)
    coordinates = np.meshgrid(*axes, indexing="ij")
    base = sum((axis + 1.0) * coordinates[axis] for axis in range(dim))

    def transient(offset: float, slope: float) -> np.ndarray:
        return np.stack((base + offset, base + offset + slope), axis=0)

    valid = np.ones(tuple(2 for _ in range(dim)), dtype=bool)
    return RegularRectilinearCompiledBackend(
        axes=axes,
        times=times,
        ux=transient(1.0, 4.0),
        uy=transient(2.0, -2.0),
        uz=transient(3.0, 1.0) if dim == 3 else None,
        electric_x=transient(10.0, 3.0),
        electric_y=transient(20.0, -1.0),
        electric_z=transient(30.0, 2.0) if dim == 3 else None,
        gas_density=transient(1.0, 0.5),
        gas_mu=transient(2.0, 0.25),
        gas_temperature=transient(300.0, 10.0),
        gas_density_source="field:rho",
        gas_mu_source="field:mu",
        gas_temperature_source="field:T",
        valid_mask=valid,
        core_valid_mask=valid,
    )


def _triangle_backend() -> TriangleMesh2DCompiledBackend:
    vertices = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    origin, cell_size, shape, offsets, indices = build_triangle_candidate_grid(
        vertices, triangles
    )
    times = np.asarray([0.0, 1.0], dtype=np.float64)
    base = vertices[:, 0] + 2.0 * vertices[:, 1]

    def quantity(name: str, offset: float, slope: float) -> QuantitySeriesND:
        return QuantitySeriesND(
            name=name,
            unit="",
            times=times,
            data=np.stack((base + offset, base + offset + slope), axis=0),
        )

    quantities = {
        "ux": quantity("ux", 1.0, 4.0),
        "uy": quantity("uy", 2.0, -2.0),
        "E_x": quantity("E_x", 10.0, 3.0),
        "E_y": quantity("E_y", 20.0, -1.0),
        "rho": quantity("rho", 1.0, 0.5),
        "mu": quantity("mu", 2.0, 0.25),
        "T": quantity("T", 300.0, 10.0),
    }
    field = TriangleMeshField2D(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        quantities=quantities,
        accel_origin=np.asarray(origin, dtype=np.float64),
        accel_cell_size=np.asarray(cell_size, dtype=np.float64),
        accel_shape=(int(shape[0]), int(shape[1])),
        accel_cell_offsets=np.asarray(offsets, dtype=np.int32),
        accel_triangle_indices=np.asarray(indices, dtype=np.int32),
        time_mode="transient",
        metadata={"field_backend_kind": "triangle_mesh_2d"},
    )
    values = {
        name: np.asarray(series.data, dtype=np.float64)
        for name, series in quantities.items()
    }
    return TriangleMesh2DCompiledBackend(
        field=field,
        velocity_names=("ux", "uy"),
        times=times,
        ux=values["ux"],
        uy=values["uy"],
        gas_density=values["rho"],
        gas_mu=values["mu"],
        gas_temperature=values["T"],
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        accel_origin=np.asarray(origin, dtype=np.float64),
        accel_cell_size=np.asarray(cell_size, dtype=np.float64),
        accel_shape=(int(shape[0]), int(shape[1])),
        accel_cell_offsets=np.asarray(offsets, dtype=np.int32),
        accel_triangle_indices=np.asarray(indices, dtype=np.int32),
        support_tolerance_m=1.0e-12,
        electric_field_names=("E_x", "E_y"),
        gas_density_source="field:rho",
        gas_mu_source="field:mu",
        gas_temperature_source="field:T",
    )


def _semantic_backend(compiled, dim: int) -> CompiledSamplingBackend:
    return CompiledSamplingBackend(
        compiled=compiled,
        spatial_dim=dim,
        fallback_density_kgm3=9.0,
        fallback_dynamic_viscosity_Pas=8.0,
        fallback_temperature_K=700.0,
    )


@pytest.mark.parametrize(
    ("compiled", "points"),
    [
        (_regular_backend(2), np.asarray([[0.2, 0.3], [0.6, 0.2]], dtype=np.float64)),
        (
            _regular_backend(3),
            np.asarray([[0.2, 0.3, 0.4], [0.6, 0.2, 0.1]], dtype=np.float64),
        ),
        (_triangle_backend(), np.asarray([[0.2, 0.3], [0.6, 0.2]], dtype=np.float64)),
    ],
    ids=("regular-2d", "regular-3d", "triangle-2d"),
)
def test_semantic_sampling_scalar_is_exactly_one_point_batch(compiled, points) -> None:
    dim = int(points.shape[1])
    backend = _semantic_backend(compiled, dim)
    batch = backend.sample(points, 0.25, ALL_QUANTITIES)

    for index, point in enumerate(points):
        scalar = sample_one(backend, point, 0.25, ALL_QUANTITIES)
        assert scalar.supported.tolist() == [bool(batch.supported[index])]
        for name in ALL_QUANTITIES.quantities:
            np.testing.assert_allclose(
                scalar.require(name)[0],
                batch.require(name)[index],
                rtol=1.0e-12,
                atol=1.0e-14,
            )


@pytest.mark.parametrize(
    ("compiled", "points"),
    [
        (_regular_backend(2), np.asarray([[0.2, 0.3], [0.6, 0.2]])),
        (
            _regular_backend(3),
            np.asarray([[0.2, 0.3, 0.4], [0.6, 0.2, 0.1]]),
        ),
        (_triangle_backend(), np.asarray([[0.2, 0.3], [0.6, 0.2]])),
    ],
    ids=("regular-2d", "regular-3d", "triangle-2d"),
)
def test_compiled_vector_sampling_preserves_float64_batch_contract(
    compiled: RegularRectilinearCompiledBackend | TriangleMesh2DCompiledBackend,
    points: np.ndarray,
) -> None:
    dim = int(points.shape[1])

    flow = sample_compiled_flow_vectors(compiled, dim, 0.25, points)
    electric = sample_compiled_electric_vectors(compiled, dim, 0.25, points)

    assert flow.shape == points.shape
    assert flow.dtype == np.float64
    assert electric is not None
    assert electric.shape == points.shape
    assert electric.dtype == np.float64


def test_compiled_vector_sampling_preserves_empty_and_missing_component_order() -> None:
    regular = replace(
        _regular_backend(3),
        electric_z=None,
    )
    missing_xy = replace(_regular_backend(2), electric_x=None)
    triangle = replace(_triangle_backend(), electric_field_names=())

    assert sample_compiled_flow_vectors(
        regular,
        3,
        0.0,
        np.empty((0, 3), dtype=np.float64),
    ).shape == (0, 3)
    assert (
        sample_compiled_electric_vectors(
            regular,
            3,
            0.0,
            np.zeros((1, 3), dtype=np.float64),
        )
        is None
    )
    empty_electric = sample_compiled_electric_vectors(
        missing_xy,
        2,
        0.0,
        np.empty((0, 2), dtype=np.float64),
    )
    assert empty_electric is not None
    assert empty_electric.shape == (0, 2)
    assert (
        sample_compiled_electric_vectors(
            missing_xy,
            2,
            0.0,
            np.zeros((1, 2), dtype=np.float64),
        )
        is None
    )
    assert (
        sample_compiled_electric_vectors(
            triangle,
            2,
            0.0,
            np.zeros((1, 2), dtype=np.float64),
        )
        is None
    )


def test_compiled_vector_validation_order_and_backend_dimension_are_stable() -> None:
    regular_2d = _regular_backend(2)
    triangle = _triangle_backend()

    with pytest.raises(ValueError, match=r"positions must have shape \(n, 2\)"):
        sample_compiled_flow_vectors(regular_2d, 2, 0.0, np.zeros(2))
    with pytest.raises(
        ValueError, match="compiled backend and requested dimension differ"
    ):
        sample_compiled_flow_vectors(regular_2d, 3, 0.0, np.zeros((1, 3)))
    with pytest.raises(
        ValueError, match="triangle mesh flow sampling is two-dimensional"
    ):
        sample_compiled_flow_vectors(triangle, 3, 0.0, np.zeros((1, 3)))
    with pytest.raises(ValueError, match="position must have shape"):
        sample_compiled_flow_vector(regular_2d, 2, 0.0, np.zeros(3))
    with pytest.raises(ValueError, match="positions must have shape"):
        sample_compiled_electric_vectors(regular_2d, 2, 0.0, np.zeros(2))


def test_compiled_gas_vector_validation_and_empty_contract_are_stable() -> None:
    regular_2d = _regular_backend(2)
    triangle = _triangle_backend()

    with pytest.raises(ValueError, match=r"positions must have shape \(n, 1\)"):
        sample_compiled_gas_properties_vectors(
            regular_2d,
            1,
            0.0,
            np.zeros((1, 1)),
            fallback_density_kgm3=9.0,
            fallback_mu_pas=8.0,
            fallback_temperature_K=700.0,
        )
    with pytest.raises(
        ValueError, match="compiled backend and requested dimension differ"
    ):
        sample_compiled_gas_properties_vectors(
            regular_2d,
            3,
            0.0,
            np.zeros((1, 3)),
            fallback_density_kgm3=9.0,
            fallback_mu_pas=8.0,
            fallback_temperature_K=700.0,
        )
    with pytest.raises(
        ValueError, match="triangle mesh gas sampling is two-dimensional"
    ):
        sample_compiled_gas_properties_vectors(
            triangle,
            3,
            0.0,
            np.zeros((1, 3)),
            fallback_density_kgm3=9.0,
            fallback_mu_pas=8.0,
            fallback_temperature_K=700.0,
        )

    density, viscosity, temperature = sample_compiled_gas_properties_vectors(
        regular_2d,
        2,
        0.0,
        np.empty((0, 2)),
        fallback_density_kgm3=9.0,
        fallback_mu_pas=8.0,
        fallback_temperature_K=700.0,
    )
    for values in (density, viscosity, temperature):
        assert values.shape == (0,)
        assert values.dtype == np.float64
    assert not np.shares_memory(density, viscosity)
    assert not np.shares_memory(viscosity, temperature)


def test_compiled_gas_vectors_keep_backend_source_and_fallback_semantics() -> None:
    points = np.asarray([[0.2, 0.3], [0.6, 0.2]], dtype=np.float64)
    regular = _regular_backend(2)
    triangle = replace(
        _triangle_backend(),
        gas_property_names={
            "gas_density": "rho",
            "gas_mu": "mu",
            "gas_temperature": "T",
        },
    )

    regular_values = sample_compiled_gas_properties_vectors(
        regular,
        2,
        0.25,
        points,
        fallback_density_kgm3=9.0,
        fallback_mu_pas=8.0,
        fallback_temperature_K=700.0,
    )
    triangle_values = sample_compiled_gas_properties_vectors(
        triangle,
        2,
        0.25,
        points,
        fallback_density_kgm3=9.0,
        fallback_mu_pas=8.0,
        fallback_temperature_K=700.0,
    )

    for regular_component, triangle_component in zip(
        regular_values, triangle_values, strict=True
    ):
        assert regular_component.dtype == np.float64
        assert triangle_component.dtype == np.float64
        np.testing.assert_allclose(
            regular_component,
            triangle_component,
            rtol=1.0e-15,
            atol=0.0,
        )


@pytest.mark.parametrize(
    ("compiled", "point"),
    [
        (_regular_backend(2), np.asarray([0.2, 0.3], dtype=np.float64)),
        (_regular_backend(3), np.asarray([0.2, 0.3, 0.4], dtype=np.float64)),
        (_triangle_backend(), np.asarray([0.2, 0.3], dtype=np.float64)),
    ],
    ids=("regular-2d", "regular-3d", "triangle-2d"),
)
def test_scalar_view_delegates_to_batch(compiled, point) -> None:
    dim = int(point.size)
    options = {
        "spatial_dim": dim,
        "need_flow": True,
        "need_electric": True,
        "need_gas_properties": True,
        "need_valid_mask": True,
        "fallback_density_kgm3": 9.0,
        "fallback_mu_pas": 8.0,
        "fallback_temperature_K": 700.0,
    }
    scalar = sample_scalar_fields_for_stage(compiled, None, point, 0.25, **options)
    batch = sample_fields_for_stage(
        compiled, None, point.reshape(1, dim), 0.25, **options
    )

    for name in (
        FLOW_VELOCITY,
        ELECTRIC_FIELD,
        GAS_DENSITY,
        DYNAMIC_VISCOSITY,
        TEMPERATURE,
        VALID_MASK_STATUS,
    ):
        np.testing.assert_allclose(
            scalar.values[name],
            batch.values[name],
            rtol=1.0e-12,
            atol=1.0e-14,
        )


def test_stage_support_is_independent_of_requested_values() -> None:
    compiled = _regular_backend(2)
    compiled.valid_mask[1, 1] = False
    backend = _semantic_backend(compiled, 2)
    sampled = backend.sample(
        np.asarray([[0.0, 0.0], [0.2, 0.2], [1.2, 0.2]], dtype=np.float64),
        0.0,
        FieldRequest((FLOW_VELOCITY,)),
    )

    assert sampled.supported.tolist() == [False, False, False]
    assert sampled.metadata["interpolation"] == "linear"


def test_triangle_backend_rejects_unsupported_3d_adapter() -> None:
    with pytest.raises(ValueError, match="dimension 2"):
        _semantic_backend(_triangle_backend(), 3)


def test_empty_value_request_still_reports_real_support() -> None:
    sampled = sample_fields_for_stage(
        _regular_backend(2),
        None,
        np.asarray([[1.2, 0.2]], dtype=np.float64),
        0.0,
        spatial_dim=2,
    )

    assert sampled.values == {}
    assert sampled.supported.tolist() == [False]


def test_explicit_empty_request_overrides_stage_plan() -> None:
    sampled = sample_fields_for_stage(
        _regular_backend(2),
        StageFieldPlan(
            need_flow=True,
            need_electric=True,
            need_gas_density=True,
            need_gas_mu=True,
            need_gas_temperature=True,
        ),
        np.asarray([[0.2, 0.2]], dtype=np.float64),
        0.0,
        spatial_dim=2,
        need_flow=False,
        need_electric=False,
        need_gas_properties=False,
        need_valid_mask=False,
    )

    assert sampled.values == {}


def test_stage_plan_gas_flags_are_used_without_explicit_override() -> None:
    sampled = sample_fields_for_stage(
        _regular_backend(2),
        StageFieldPlan(
            need_flow=False,
            need_gas_density=True,
            need_gas_temperature=True,
            need_valid_mask=False,
        ),
        np.asarray([[0.2, 0.2]], dtype=np.float64),
        0.0,
        spatial_dim=2,
    )

    assert tuple(sampled.values) == (GAS_DENSITY, TEMPERATURE)


@pytest.mark.parametrize(
    ("points", "spatial_dim"),
    [
        (np.asarray([0.2, 0.2]), 2),
        (np.asarray([[0.2, 0.2]]), 3),
    ],
)
def test_stage_sampling_rejects_invalid_batch_shape(
    points: np.ndarray,
    spatial_dim: int,
) -> None:
    with pytest.raises(ValueError, match=r"points must have shape \(n, spatial_dim\)"):
        sample_fields_for_stage(
            _regular_backend(2),
            None,
            points,
            0.0,
            spatial_dim=spatial_dim,
        )


def test_scalar_stage_sampling_rejects_invalid_shape() -> None:
    with pytest.raises(
        ValueError,
        match=r"position must have shape \(spatial_dim,\)",
    ):
        sample_scalar_fields_for_stage(
            _regular_backend(2),
            None,
            np.asarray([[0.2, 0.2]]),
            0.0,
            spatial_dim=2,
        )


def test_declared_invalid_gas_field_is_not_repaired_by_config_fallback() -> None:
    compiled = _regular_backend(2)
    compiled = replace(
        compiled,
        gas_temperature=np.zeros_like(compiled.gas_temperature),
    )
    backend = _semantic_backend(compiled, 2)

    with pytest.raises(
        ValueError,
        match=(
            "temperature was requested but neither the field nor an explicit gas "
            "fallback"
        ),
    ):
        backend.sample(
            np.asarray([[0.5, 0.5]], dtype=np.float64),
            0.25,
            FieldRequest((TEMPERATURE,)),
        )


def test_manifest_normalized_components_are_the_only_runtime_quantity_names(
    tmp_path,
) -> None:
    axes = np.asarray([0.0, 1.0], dtype=np.float64)
    raw_x = np.full((2, 2, 2), 20.0, dtype=np.float64)
    raw_y = np.full((2, 2, 2), -30.0, dtype=np.float64)
    field_path = tmp_path / "field.npz"
    np.savez(
        field_path,
        axis_0=axes,
        axis_1=axes,
        times=np.asarray([0.0, 1.0], dtype=np.float64),
        valid_mask=np.ones((2, 2), dtype=bool),
        arbitrary_component_a=raw_y,
        arbitrary_component_b=raw_x,
    )
    provider = build_precomputed_field(
        {
            "npz_path": str(field_path),
            "strict_quantity_mapping": True,
            "quantity_mapping": {
                "ux": {
                    "source": "arbitrary_component_b",
                    "unit": "m/s",
                    "scale_to_si": 0.1,
                    "semantic_quantity": "velocity",
                    "component": "x",
                },
                "uy": {
                    "source": "arbitrary_component_a",
                    "unit": "m/s",
                    "scale_to_si": 0.1,
                    "semantic_quantity": "velocity",
                    "component": "y",
                },
            },
        },
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        axes=(axes, axes),
    )
    runtime = SimpleNamespace(
        geometry_provider=SimpleNamespace(
            geometry=SimpleNamespace(
                axes=(axes, axes), valid_mask=np.ones((2, 2), dtype=bool)
            )
        ),
        field_provider=provider,
        gas=SimpleNamespace(
            density_kgm3=1.0, dynamic_viscosity_Pas=1.0, temperature=300.0
        ),
    )
    compiled = compile_runtime_backend(runtime, 2)
    sampled = _semantic_backend(compiled, 2).sample(
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        0.5,
        FieldRequest((FLOW_VELOCITY,)),
    )

    assert set(provider.field.quantities) == {"ux", "uy"}
    np.testing.assert_allclose(
        sampled.require(FLOW_VELOCITY),
        np.asarray([[2.0, -3.0]], dtype=np.float64),
        rtol=1.0e-12,
        atol=0.0,
    )
