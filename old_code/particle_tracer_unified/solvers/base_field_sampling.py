"""Sample compiled backend fields without assembling force-specific terms."""

from __future__ import annotations

import numpy as np

from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
    sample_time_grid_scalar,
    sample_valid_mask_status,
)
from particle_tracer_unified.core.triangle_mesh_sampling_2d import (
    locate_triangle_containing_point,
    sample_triangle_mesh_status,
)

from ._field_sampling_report import compiled_gas_property_report
from .compiled_backend_types import (
    CompiledRuntimeBackend,
    RegularRectilinearCompiledBackend,
    TriangleMesh2DCompiledBackend,
)
from .triangle_derived_fields import (
    triangle_sample_error,
    triangle_series_gradient_at_location,
    triangle_series_time_derivative_at_location,
    triangle_series_value_at_location,
)


def _axis_intervals(
    axis: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(axis, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    hi = np.searchsorted(arr, vals, side="right")
    hi = np.clip(hi, 1, arr.size - 1).astype(np.int64)
    lo = hi - 1
    denominator = arr[hi] - arr[lo]
    if np.any(~np.isfinite(denominator)) or np.any(denominator <= 0.0):
        raise ValueError("Compiled field axes must be finite and strictly increasing")
    alpha = (vals - arr[lo]) / denominator
    alpha = np.where(vals <= arr[0], 0.0, np.where(vals >= arr[-1], 1.0, alpha))
    return lo, hi, np.clip(alpha, 0.0, 1.0)


def _sample_regular_grid_points_2d(
    grid: np.ndarray,
    axes: tuple[np.ndarray, ...],
    positions: np.ndarray,
) -> np.ndarray:
    points = np.asarray(positions, dtype=np.float64)
    data = np.asarray(grid, dtype=np.float64)
    ix0, ix1, ax = _axis_intervals(axes[0], points[:, 0])
    iy0, iy1, ay = _axis_intervals(axes[1], points[:, 1])
    c00 = data[ix0, iy0]
    c10 = data[ix1, iy0]
    c01 = data[ix0, iy1]
    c11 = data[ix1, iy1]
    c0 = c00 * (1.0 - ax) + c10 * ax
    c1 = c01 * (1.0 - ax) + c11 * ax
    return c0 * (1.0 - ay) + c1 * ay


def sample_regular_time_grid_points_2d(
    data: np.ndarray,
    axes: tuple[np.ndarray, ...],
    times: np.ndarray,
    t_eval: float,
    positions: np.ndarray,
) -> np.ndarray:
    """Vectorized bilinear/linear sampling for the 2-D regular hot path."""

    arr = np.asarray(data, dtype=np.float64)
    time_grid = np.asarray(times, dtype=np.float64)
    if arr.ndim == 2:
        return _sample_regular_grid_points_2d(arr, axes, positions)
    if arr.shape[0] <= 1 or time_grid.size <= 1:
        return _sample_regular_grid_points_2d(arr[0], axes, positions)
    if t_eval <= float(time_grid[0]):
        return _sample_regular_grid_points_2d(arr[0], axes, positions)
    if t_eval >= float(time_grid[-1]):
        return _sample_regular_grid_points_2d(arr[-1], axes, positions)
    hi = int(np.searchsorted(time_grid, float(t_eval)))
    lo = hi - 1
    t_lo = float(time_grid[lo])
    interval = float(time_grid[hi]) - t_lo
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("Compiled field times must be finite and strictly increasing")
    alpha = (float(t_eval) - t_lo) / interval
    v_lo = _sample_regular_grid_points_2d(arr[lo], axes, positions)
    v_hi = _sample_regular_grid_points_2d(arr[hi], axes, positions)
    return v_lo * (1.0 - alpha) + v_hi * alpha


def sample_regular_time_grid_points(
    backend: RegularRectilinearCompiledBackend,
    data: np.ndarray,
    t_eval: float,
    points: np.ndarray,
) -> np.ndarray:
    if len(backend.axes) == 2:
        return sample_regular_time_grid_points_2d(
            data,
            backend.axes,
            backend.times,
            float(t_eval),
            points,
        )
    return np.asarray(
        [
            sample_time_grid_scalar(
                data,
                backend.axes,
                backend.times,
                float(t_eval),
                point,
            )
            for point in points
        ],
        dtype=np.float64,
    )


def sample_regular_components(
    backend: RegularRectilinearCompiledBackend,
    components: tuple[np.ndarray | None, ...],
    t_eval: float,
    points: np.ndarray,
) -> np.ndarray | None:
    if any(component is None for component in components):
        return None
    return np.column_stack(
        tuple(
            sample_regular_time_grid_points(backend, component, float(t_eval), points)
            for component in components
            if component is not None
        )
    ).astype(np.float64, copy=False)


def sample_regular_velocity_gradient(
    backend: RegularRectilinearCompiledBackend,
    spatial_dim: int,
    t_eval: float,
    points: np.ndarray,
) -> np.ndarray | None:
    components = (
        (
            (backend.grad_ux_x, backend.grad_ux_y)
            if int(spatial_dim) == 2
            else (backend.grad_ux_x, backend.grad_ux_y, backend.grad_ux_z)
        ),
        (
            (backend.grad_uy_x, backend.grad_uy_y)
            if int(spatial_dim) == 2
            else (backend.grad_uy_x, backend.grad_uy_y, backend.grad_uy_z)
        ),
    )
    if int(spatial_dim) == 3:
        components += ((backend.grad_uz_x, backend.grad_uz_y, backend.grad_uz_z),)
    if any(value is None for row in components for value in row):
        return None
    sampled = np.empty(
        (points.shape[0], int(spatial_dim), int(spatial_dim)),
        dtype=np.float64,
    )
    for row_index, row in enumerate(components):
        for column_index, value in enumerate(row):
            sampled[:, row_index, column_index] = sample_regular_time_grid_points(
                backend,
                value,
                float(t_eval),
                points,
            )
    return sampled


def triangle_mesh_location(
    backend: TriangleMesh2DCompiledBackend,
    position: np.ndarray,
) -> tuple[int, np.ndarray]:
    return locate_triangle_containing_point(
        vertices=backend.mesh_vertices,
        triangles=backend.mesh_triangles,
        accel_origin=backend.accel_origin,
        accel_cell_size=backend.accel_cell_size,
        accel_shape=backend.accel_shape,
        accel_cell_offsets=backend.accel_cell_offsets,
        accel_triangle_indices=backend.accel_triangle_indices,
        position=np.asarray(position, dtype=np.float64),
        eps=float(backend.support_tolerance_m),
        nearest_fallback=True,
    )


def sample_triangle_velocity_terms(
    backend: TriangleMesh2DCompiledBackend,
    t_eval: float,
    position: np.ndarray,
    *,
    row_index: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field = backend.field
    names = backend.velocity_names
    if len(names) < 2:
        return (
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
            np.zeros((2, 2), dtype=np.float64),
        )
    triangle_index, barycentric = triangle_mesh_location(backend, position)
    if int(triangle_index) < 0:
        return (
            np.full(2, np.nan, dtype=np.float64),
            np.full(2, np.nan, dtype=np.float64),
            np.full((2, 2), np.nan, dtype=np.float64),
        )
    ux_series = field.quantities[names[0]]
    uy_series = field.quantities[names[1]]
    flow = np.asarray(
        [
            triangle_series_value_at_location(
                ux_series,
                field,
                int(triangle_index),
                barycentric,
                float(t_eval),
                semantic_quantity="flow_velocity.x",
                row_index=row_index,
            ),
            triangle_series_value_at_location(
                uy_series,
                field,
                int(triangle_index),
                barycentric,
                float(t_eval),
                semantic_quantity="flow_velocity.y",
                row_index=row_index,
            ),
        ],
        dtype=np.float64,
    )
    time_derivative = np.asarray(
        [
            triangle_series_time_derivative_at_location(
                ux_series,
                field,
                int(triangle_index),
                barycentric,
                float(t_eval),
                semantic_quantity="flow_time_derivative.x",
                row_index=row_index,
            ),
            triangle_series_time_derivative_at_location(
                uy_series,
                field,
                int(triangle_index),
                barycentric,
                float(t_eval),
                semantic_quantity="flow_time_derivative.y",
                row_index=row_index,
            ),
        ],
        dtype=np.float64,
    )
    gradient = np.vstack(
        (
            triangle_series_gradient_at_location(
                ux_series,
                field,
                int(triangle_index),
                float(t_eval),
                semantic_quantity="flow_velocity_gradient.x",
                row_index=row_index,
            ),
            triangle_series_gradient_at_location(
                uy_series,
                field,
                int(triangle_index),
                float(t_eval),
                semantic_quantity="flow_velocity_gradient.y",
                row_index=row_index,
            ),
        )
    ).astype(np.float64, copy=False)
    return flow, time_derivative, gradient


def sample_triangle_scalar_gradient(
    backend: TriangleMesh2DCompiledBackend,
    quantity_name: str,
    t_eval: float,
    position: np.ndarray,
    *,
    semantic_quantity: str,
    row_index: int | None,
) -> np.ndarray:
    field = backend.field
    series = field.quantities.get(str(quantity_name))
    if series is None:
        return np.zeros(2, dtype=np.float64)
    triangle_index, _barycentric = triangle_mesh_location(backend, position)
    if int(triangle_index) < 0:
        return np.full(2, np.nan, dtype=np.float64)
    return triangle_series_gradient_at_location(
        series,
        field,
        int(triangle_index),
        float(t_eval),
        semantic_quantity=semantic_quantity,
        row_index=row_index,
    )


def sample_triangle_scalar_value(
    backend: TriangleMesh2DCompiledBackend,
    quantity_name: str,
    t_eval: float,
    position: np.ndarray,
    *,
    semantic_quantity: str,
    row_index: int | None,
) -> float:
    field = backend.field
    series = field.quantities.get(str(quantity_name))
    if series is None:
        return 0.0
    triangle_index, barycentric = triangle_mesh_location(backend, position)
    if int(triangle_index) < 0:
        return float("nan")
    return triangle_series_value_at_location(
        series,
        field,
        int(triangle_index),
        barycentric,
        float(t_eval),
        semantic_quantity=semantic_quantity,
        row_index=row_index,
    )


def sample_triangle_vector(
    backend: TriangleMesh2DCompiledBackend,
    quantity_names: tuple[str, ...],
    semantic_quantity: str,
    t_eval: float,
    points: np.ndarray,
) -> np.ndarray:
    field = backend.field
    result = np.full((points.shape[0], len(quantity_names)), np.nan, dtype=np.float64)
    for row_index, point in enumerate(points):
        triangle_index, barycentric = triangle_mesh_location(backend, point)
        if int(triangle_index) < 0:
            continue
        for component_index, quantity_name in enumerate(quantity_names):
            series = field.quantities.get(str(quantity_name))
            if series is None:
                raise triangle_sample_error(
                    f"{semantic_quantity}.{component_index}",
                    f"declared component '{quantity_name}' is unavailable",
                    row_index=row_index,
                    triangle_index=int(triangle_index),
                )
            component = (
                "xyz"[component_index] if component_index < 3 else str(component_index)
            )
            result[row_index, component_index] = triangle_series_value_at_location(
                series,
                field,
                int(triangle_index),
                barycentric,
                float(t_eval),
                semantic_quantity=f"{semantic_quantity}.{component}",
                row_index=row_index,
            )
    return result


def _sample_regular_vector(
    compiled: RegularRectilinearCompiledBackend,
    components: tuple[np.ndarray, ...],
    t_eval: float,
    points: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        tuple(
            sample_regular_time_grid_points(
                compiled,
                component,
                float(t_eval),
                points,
            )
            for component in components
        )
    ).astype(np.float64, copy=False)


def _sample_regular_flow_vectors(
    compiled: RegularRectilinearCompiledBackend,
    spatial_dim: int,
    t_eval: float,
    points: np.ndarray,
) -> np.ndarray:
    if len(compiled.axes) != spatial_dim:
        raise ValueError("compiled backend and requested dimension differ")
    components = (compiled.ux, compiled.uy)
    if spatial_dim == 3:
        z_component = (
            compiled.uz
            if compiled.uz is not None
            else np.zeros_like(compiled.ux, dtype=np.float64)
        )
        components += (z_component,)
    values = _sample_regular_vector(compiled, components, t_eval, points)
    statuses = sample_compiled_valid_mask_statuses(compiled, points)
    invalid_rows = np.flatnonzero(
        (statuses == np.uint8(VALID_MASK_STATUS_CLEAN))
        & np.any(~np.isfinite(values), axis=1)
    )
    if invalid_rows.size:
        raise ValueError(
            "regular field semantic quantity 'flow_velocity' contains non-finite "
            f"samples at rows {invalid_rows.tolist()}"
        )
    return values


def _regular_electric_components(
    compiled: RegularRectilinearCompiledBackend,
    spatial_dim: int,
) -> tuple[np.ndarray, ...] | None:
    if compiled.electric_x is None or compiled.electric_y is None:
        return None
    if spatial_dim == 3 and compiled.electric_z is None:
        return None
    components = (
        (compiled.electric_x, compiled.electric_y)
        if spatial_dim == 2
        else (compiled.electric_x, compiled.electric_y, compiled.electric_z)
    )
    return tuple(component for component in components if component is not None)


def sample_compiled_flow_vector(
    compiled: CompiledRuntimeBackend,
    spatial_dim: int,
    t_eval: float,
    position: np.ndarray,
) -> np.ndarray:
    point = np.asarray(position, dtype=np.float64)
    dim = int(spatial_dim)
    if point.shape != (dim,):
        raise ValueError(f"position must have shape ({dim},)")
    return sample_compiled_flow_vectors(
        compiled, dim, float(t_eval), point.reshape(1, dim)
    )[0]


def sample_compiled_flow_vectors(
    compiled: CompiledRuntimeBackend,
    spatial_dim: int,
    t_eval: float,
    positions: np.ndarray,
) -> np.ndarray:
    points = np.asarray(positions, dtype=np.float64)
    dim = int(spatial_dim)
    if dim not in (2, 3) or points.ndim != 2 or points.shape[1] != dim:
        raise ValueError(f"positions must have shape (n, {dim})")
    if points.shape[0] == 0:
        return np.zeros((0, dim), dtype=np.float64)
    if isinstance(compiled, TriangleMesh2DCompiledBackend):
        if dim != 2:
            raise ValueError("triangle mesh flow sampling is two-dimensional")
        if not compiled.velocity_names:
            return np.zeros((points.shape[0], dim), dtype=np.float64)
        values = sample_triangle_vector(
            compiled,
            tuple(compiled.velocity_names),
            "flow_velocity",
            float(t_eval),
            points,
        )
    else:
        values = _sample_regular_flow_vectors(compiled, dim, float(t_eval), points)
    return np.asarray(values, dtype=np.float64)


def sample_compiled_electric_vectors(
    compiled: CompiledRuntimeBackend,
    spatial_dim: int,
    t_eval: float,
    positions: np.ndarray,
) -> np.ndarray | None:
    points = np.asarray(positions, dtype=np.float64)
    dim = int(spatial_dim)
    if points.ndim != 2 or points.shape[1] != dim:
        raise ValueError("positions must have shape (n, spatial_dim)")
    if points.shape[0] == 0:
        return np.zeros((0, dim), dtype=np.float64)
    if isinstance(compiled, TriangleMesh2DCompiledBackend):
        names = tuple(compiled.electric_field_names)
        if dim != 2 or len(names) != 2:
            return None
        return sample_triangle_vector(
            compiled, names, "electric_field", float(t_eval), points
        )
    components = _regular_electric_components(compiled, dim)
    if components is None:
        return None
    return _sample_regular_vector(compiled, components, float(t_eval), points)


def sample_compiled_gas_properties(
    compiled: CompiledRuntimeBackend,
    t_eval: float,
    position: np.ndarray,
    *,
    fallback_density_kgm3: float,
    fallback_mu_pas: float,
    fallback_temperature_K: float,
) -> tuple[float, float, float]:
    dim = (
        2 if isinstance(compiled, TriangleMesh2DCompiledBackend) else len(compiled.axes)
    )
    point = np.asarray(position, dtype=np.float64)
    if point.shape != (dim,):
        raise ValueError(f"position must have shape ({dim},)")
    density, viscosity, temperature = sample_compiled_gas_properties_vectors(
        compiled,
        dim,
        float(t_eval),
        point.reshape(1, dim),
        fallback_density_kgm3=float(fallback_density_kgm3),
        fallback_mu_pas=float(fallback_mu_pas),
        fallback_temperature_K=float(fallback_temperature_K),
    )
    return float(density[0]), float(viscosity[0]), float(temperature[0])


def _triangle_gas_property_samples(
    compiled: TriangleMesh2DCompiledBackend,
    t_eval: float,
    points: np.ndarray,
    fallback_values: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(points.shape[0])
    sampled = {
        "gas_density": np.full(count, fallback_values[0], dtype=np.float64),
        "gas_mu": np.full(count, fallback_values[1], dtype=np.float64),
        "gas_temperature": np.full(count, fallback_values[2], dtype=np.float64),
    }
    for target, name in compiled.gas_property_names.items():
        sampled[target] = sample_triangle_vector(
            compiled,
            (name,),
            target,
            float(t_eval),
            points,
        )[:, 0]
    return sampled["gas_density"], sampled["gas_mu"], sampled["gas_temperature"]


def _regular_gas_property_samples(
    compiled: RegularRectilinearCompiledBackend,
    t_eval: float,
    points: np.ndarray,
    fallback_values: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sampled: list[np.ndarray] = []
    for data, source, fallback_value in (
        (compiled.gas_density, compiled.gas_density_source, fallback_values[0]),
        (compiled.gas_mu, compiled.gas_mu_source, fallback_values[1]),
        (
            compiled.gas_temperature,
            compiled.gas_temperature_source,
            fallback_values[2],
        ),
    ):
        values = sample_regular_time_grid_points(compiled, data, float(t_eval), points)
        if not str(source).startswith("field:"):
            values = np.where(
                np.isfinite(values) & (values > 0.0),
                values,
                fallback_value,
            )
        sampled.append(values.astype(np.float64, copy=False))
    return sampled[0], sampled[1], sampled[2]


def sample_compiled_gas_properties_vectors(
    compiled: CompiledRuntimeBackend,
    spatial_dim: int,
    t_eval: float,
    positions: np.ndarray,
    *,
    fallback_density_kgm3: float,
    fallback_mu_pas: float,
    fallback_temperature_K: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(positions, dtype=np.float64)
    dim = int(spatial_dim)
    if dim not in (2, 3) or points.ndim != 2 or points.shape[1] != dim:
        raise ValueError(f"positions must have shape (n, {dim})")
    count = int(points.shape[0])
    if count == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    fallback_values = (
        float(fallback_density_kgm3),
        float(fallback_mu_pas),
        float(fallback_temperature_K),
    )
    if isinstance(compiled, TriangleMesh2DCompiledBackend):
        if dim != 2:
            raise ValueError("triangle mesh gas sampling is two-dimensional")
        return _triangle_gas_property_samples(
            compiled,
            float(t_eval),
            points,
            fallback_values,
        )
    if len(compiled.axes) != dim:
        raise ValueError("compiled backend and requested dimension differ")
    return _regular_gas_property_samples(
        compiled,
        float(t_eval),
        points,
        fallback_values,
    )


def sample_compiled_valid_mask_status(
    compiled: CompiledRuntimeBackend,
    position: np.ndarray,
) -> int:
    dim = (
        2 if isinstance(compiled, TriangleMesh2DCompiledBackend) else len(compiled.axes)
    )
    point = np.asarray(position, dtype=np.float64)
    if point.shape != (dim,):
        raise ValueError(f"position must have shape ({dim},)")
    return int(sample_compiled_valid_mask_statuses(compiled, point.reshape(1, dim))[0])


def sample_compiled_valid_mask_statuses(
    compiled: CompiledRuntimeBackend,
    positions: np.ndarray,
) -> np.ndarray:
    points = np.asarray(positions, dtype=np.float64)
    dim = (
        2 if isinstance(compiled, TriangleMesh2DCompiledBackend) else len(compiled.axes)
    )
    if points.ndim != 2 or points.shape[1] != dim:
        raise ValueError(f"positions must have shape (n, {dim})")
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.uint8)
    if isinstance(compiled, TriangleMesh2DCompiledBackend):
        return np.asarray(
            [sample_triangle_mesh_status(compiled.field, point) for point in points],
            dtype=np.uint8,
        )
    if len(compiled.axes) == 2:
        axes = compiled.axes
        mask = np.asarray(compiled.valid_mask, dtype=bool)
        inside = (
            np.all(np.isfinite(points[:, :2]), axis=1)
            & (points[:, 0] >= float(axes[0][0]))
            & (points[:, 0] <= float(axes[0][-1]))
            & (points[:, 1] >= float(axes[1][0]))
            & (points[:, 1] <= float(axes[1][-1]))
        )
        ix0, ix1, _ax = _axis_intervals(axes[0], points[:, 0])
        iy0, iy1, _ay = _axis_intervals(axes[1], points[:, 1])
        point_values = _sample_regular_grid_points_2d(
            mask.astype(np.float64), axes, points
        )
        point_valid = inside & (point_values >= 0.5)
        stencil_invalid = (
            (~inside)
            | (~mask[ix0, iy0])
            | (~mask[ix1, iy0])
            | (~mask[ix0, iy1])
            | (~mask[ix1, iy1])
        )
        statuses = np.full(
            points.shape[0], int(VALID_MASK_STATUS_CLEAN), dtype=np.uint8
        )
        statuses[stencil_invalid] = np.uint8(VALID_MASK_STATUS_MIXED_STENCIL)
        statuses[~point_valid] = np.uint8(VALID_MASK_STATUS_HARD_INVALID)
        return statuses
    return np.asarray(
        [
            sample_valid_mask_status(
                np.asarray(compiled.valid_mask, dtype=bool),
                compiled.axes,
                point,
            )
            for point in points
        ],
        dtype=np.uint8,
    )


__all__ = (
    "compiled_gas_property_report",
    "sample_compiled_electric_vectors",
    "sample_compiled_flow_vector",
    "sample_compiled_flow_vectors",
    "sample_compiled_gas_properties",
    "sample_compiled_gas_properties_vectors",
    "sample_compiled_valid_mask_status",
    "sample_compiled_valid_mask_statuses",
    "sample_regular_components",
    "sample_regular_time_grid_points",
    "sample_regular_time_grid_points_2d",
    "sample_regular_velocity_gradient",
    "sample_triangle_scalar_gradient",
    "sample_triangle_scalar_value",
    "sample_triangle_vector",
    "sample_triangle_velocity_terms",
)
