from __future__ import annotations

import numpy as np

from particle_tracer_unified.core.datamodel import TriangleMeshField2D


def triangle_sample_error(
    semantic_quantity: str,
    detail: str,
    *,
    row_index: int | None,
    triangle_index: int | None,
) -> ValueError:
    location = []
    if row_index is not None:
        location.append(f"row {int(row_index)}")
    if triangle_index is not None:
        location.append(f"triangle {int(triangle_index)}")
    suffix = f" at {', '.join(location)}" if location else ""
    return ValueError(
        f"triangle field semantic quantity '{semantic_quantity}'{suffix}: {detail}"
    )


def _time_interval_roundoff_s(t_lo: float, t_hi: float) -> float:
    ulp_lo = abs(float(np.spacing(np.float64(t_lo))))
    ulp_hi = abs(float(np.spacing(np.float64(t_hi))))
    return 64.0 * max(ulp_lo, ulp_hi, float(np.nextafter(0.0, 1.0)))


def _series_time_count(data: np.ndarray, vertex_count: int) -> int | None:
    if data.ndim == 1:
        return 1 if data.shape == (vertex_count,) else None
    if data.ndim == 2 and data.shape[1] == vertex_count and data.shape[0] >= 1:
        return int(data.shape[0])
    return None


def _time_axis_error(times: np.ndarray, time_count: int, name: str) -> str | None:
    if times.ndim != 1 or times.size != time_count:
        return f"quantity '{name}' time axis does not match its {time_count} data rows"
    if not np.all(np.isfinite(times)):
        return f"quantity '{name}' time axis contains non-finite values"
    for time_index in range(max(0, int(times.size) - 1)):
        t_lo = float(times[time_index])
        t_hi = float(times[time_index + 1])
        dt = t_hi - t_lo
        if not np.isfinite(dt) or dt <= _time_interval_roundoff_s(t_lo, t_hi):
            return (
                f"quantity '{name}' has an unresolved float64 time interval "
                f"between time rows {time_index} and {time_index + 1}"
            )
    return None


def _series_contract(
    series,
    field: TriangleMeshField2D,
    semantic_quantity: str,
    *,
    row_index: int | None,
    triangle_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(series.data, dtype=np.float64)
    times = np.asarray(series.times, dtype=np.float64)
    vertex_count = int(np.asarray(field.mesh_vertices).shape[0])
    time_count = _series_time_count(data, vertex_count)
    if time_count is None:
        raise triangle_sample_error(
            semantic_quantity,
            (
                f"quantity '{series.name}' data must have shape "
                "(vertex,) or (time, vertex)"
            ),
            row_index=row_index,
            triangle_index=triangle_index,
        )
    time_error = _time_axis_error(times, time_count, series.name)
    if time_error is not None:
        raise triangle_sample_error(
            semantic_quantity,
            time_error,
            row_index=row_index,
            triangle_index=triangle_index,
        )
    return data, times


def _geometry_terms(
    field: TriangleMeshField2D,
    triangle_index: int,
    semantic_quantity: str,
    *,
    row_index: int | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    tri = np.asarray(field.mesh_triangles, dtype=np.int32)[int(triangle_index)]
    points = np.asarray(field.mesh_vertices, dtype=np.float64)[tri]
    if not np.all(np.isfinite(points)):
        raise triangle_sample_error(
            semantic_quantity,
            "triangle vertices contain non-finite coordinates",
            row_index=row_index,
            triangle_index=triangle_index,
        )
    edge_1 = points[1] - points[0]
    edge_2 = points[2] - points[0]
    edge_scale = float(max(np.max(np.abs(edge_1)), np.max(np.abs(edge_2))))
    if not np.isfinite(edge_scale) or edge_scale <= 0.0:
        raise triangle_sample_error(
            semantic_quantity,
            "triangle has no resolvable positive edge scale",
            row_index=row_index,
            triangle_index=triangle_index,
        )
    normalized_1 = edge_1 / edge_scale
    normalized_2 = edge_2 / edge_scale
    gram_scale = float(np.dot(normalized_1, normalized_1)) * float(
        np.dot(normalized_2, normalized_2)
    )
    determinant = float(
        normalized_1[0] * normalized_2[1] - normalized_1[1] * normalized_2[0]
    )
    # det([e1,e2])**2 is the 2D edge Gram determinant.  The cross-product
    # form avoids cancellation in g11*g22-g12**2 for slender triangles.
    gram_determinant = determinant * determinant
    relative_limit = (64.0 * np.finfo(np.float64).eps) ** 2 * gram_scale
    if (
        not np.isfinite(gram_determinant)
        or gram_scale <= 0.0
        or gram_determinant <= relative_limit
    ):
        raise triangle_sample_error(
            semantic_quantity,
            "triangle edge Gram determinant is unresolved in float64",
            row_index=row_index,
            triangle_index=triangle_index,
        )
    return normalized_1, normalized_2, determinant, edge_scale


def validate_triangle_gradient_geometry(
    field: TriangleMeshField2D,
    semantic_quantity: str,
) -> None:
    for triangle_index in range(int(np.asarray(field.mesh_triangles).shape[0])):
        _geometry_terms(
            field,
            triangle_index,
            semantic_quantity,
            row_index=None,
        )


def _values_at_time(
    series,
    field: TriangleMeshField2D,
    triangle_index: int,
    time_s: float,
    *,
    semantic_quantity: str,
    row_index: int | None,
) -> np.ndarray:
    tri = np.asarray(field.mesh_triangles, dtype=np.int32)[int(triangle_index)]
    data, times = _series_contract(
        series,
        field,
        semantic_quantity,
        row_index=row_index,
        triangle_index=int(triangle_index),
    )
    if not np.isfinite(float(time_s)):
        raise triangle_sample_error(
            semantic_quantity,
            "sample time is non-finite",
            row_index=row_index,
            triangle_index=triangle_index,
        )
    if data.ndim == 1:
        values = np.asarray(data[tri], dtype=np.float64)
    elif data.shape[0] == 1 or float(time_s) <= float(times[0]):
        values = np.asarray(data[0, tri], dtype=np.float64)
    elif float(time_s) >= float(times[-1]):
        values = np.asarray(data[-1, tri], dtype=np.float64)
    else:
        hi = int(np.searchsorted(times, float(time_s)))
        lo = hi - 1
        alpha = (float(time_s) - float(times[lo])) / (
            float(times[hi]) - float(times[lo])
        )
        values = np.asarray(
            data[lo, tri] * (1.0 - alpha) + data[hi, tri] * alpha,
            dtype=np.float64,
        )
    if not np.all(np.isfinite(values)):
        raise triangle_sample_error(
            semantic_quantity,
            f"quantity '{series.name}' produced non-finite vertex values",
            row_index=row_index,
            triangle_index=triangle_index,
        )
    return values


def triangle_series_value_at_location(
    series,
    field: TriangleMeshField2D,
    triangle_index: int,
    barycentric: np.ndarray,
    time_s: float,
    *,
    semantic_quantity: str,
    row_index: int | None,
) -> float:
    values = _values_at_time(
        series,
        field,
        int(triangle_index),
        float(time_s),
        semantic_quantity=semantic_quantity,
        row_index=row_index,
    )
    value = float(np.dot(np.asarray(barycentric, dtype=np.float64), values))
    if not np.isfinite(value):
        raise triangle_sample_error(
            semantic_quantity,
            f"quantity '{series.name}' produced a non-finite interpolated value",
            row_index=row_index,
            triangle_index=triangle_index,
        )
    return value


def triangle_series_time_derivative_at_location(
    series,
    field: TriangleMeshField2D,
    triangle_index: int,
    barycentric: np.ndarray,
    time_s: float,
    *,
    semantic_quantity: str,
    row_index: int | None,
) -> float:
    tri = np.asarray(field.mesh_triangles, dtype=np.int32)[int(triangle_index)]
    if not np.isfinite(float(time_s)):
        raise triangle_sample_error(
            semantic_quantity,
            "sample time is non-finite",
            row_index=row_index,
            triangle_index=triangle_index,
        )
    data, times = _series_contract(
        series,
        field,
        semantic_quantity,
        row_index=row_index,
        triangle_index=int(triangle_index),
    )
    if data.ndim == 1 or data.shape[0] == 1:
        return 0.0
    if float(time_s) <= float(times[0]):
        lo, hi = 0, 1
    elif float(time_s) >= float(times[-1]):
        lo, hi = int(times.size) - 2, int(times.size) - 1
    else:
        hi = int(np.searchsorted(times, float(time_s)))
        lo = hi - 1
    weights = np.asarray(barycentric, dtype=np.float64)
    v_lo = float(np.dot(weights, data[lo, tri]))
    v_hi = float(np.dot(weights, data[hi, tri]))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        derivative = float((v_hi - v_lo) / (float(times[hi]) - float(times[lo])))
    if not np.isfinite(v_lo) or not np.isfinite(v_hi) or not np.isfinite(derivative):
        raise triangle_sample_error(
            semantic_quantity,
            f"quantity '{series.name}' produced a non-finite time derivative",
            row_index=row_index,
            triangle_index=triangle_index,
        )
    return derivative


def triangle_series_gradient_at_location(
    series,
    field: TriangleMeshField2D,
    triangle_index: int,
    time_s: float,
    *,
    semantic_quantity: str,
    row_index: int | None,
) -> np.ndarray:
    values = _values_at_time(
        series,
        field,
        int(triangle_index),
        float(time_s),
        semantic_quantity=semantic_quantity,
        row_index=row_index,
    )
    edge_1, edge_2, determinant, edge_scale = _geometry_terms(
        field,
        int(triangle_index),
        semantic_quantity,
        row_index=row_index,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        dv1 = float(values[1] - values[0])
        dv2 = float(values[2] - values[0])
        gradient = np.asarray(
            [
                (edge_2[1] * dv1 - edge_1[1] * dv2) / determinant / edge_scale,
                (-edge_2[0] * dv1 + edge_1[0] * dv2) / determinant / edge_scale,
            ],
            dtype=np.float64,
        )
    if not np.all(np.isfinite(gradient)):
        raise triangle_sample_error(
            semantic_quantity,
            f"quantity '{series.name}' produced a non-finite spatial gradient",
            row_index=row_index,
            triangle_index=triangle_index,
        )
    return gradient


__all__ = (
    "triangle_sample_error",
    "triangle_series_gradient_at_location",
    "triangle_series_time_derivative_at_location",
    "triangle_series_value_at_location",
    "validate_triangle_gradient_geometry",
)
