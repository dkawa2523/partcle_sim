"""Build regular-grid field providers from precomputed arrays."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from particle_tracer_unified.core.coordinate_systems import (
    axis_names_for_coordinate_system,
)
from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    QuantitySeriesND,
    RegularFieldND,
)
from particle_tracer_unified.core.numerical_contracts import float_arrays_equal_ulps

from ._precomputed_common import (
    coordinate_scale,
    infer_unit,
    quantity_mapping,
    quantity_metadata,
    quantity_sources,
    read_axes,
    read_metadata,
    read_times,
    real_quantity_values,
    resolve_path,
)

_REGULAR_RESERVED_ARRAYS = {
    "axis_0",
    "axis_1",
    "axis_2",
    "times",
    "valid_mask",
    "support_phi",
    "metadata_json",
    "sdf",
    "part_id_map",
    "nearest_boundary_part_id_map",
    "normal_0",
    "normal_1",
    "normal_2",
    "boundary_edges",
    "boundary_edge_part_ids",
    "boundary_triangles",
    "boundary_triangle_part_ids",
    "containment_boundary_triangles",
    "boundary_loops_2d_flat",
    "boundary_loops_2d_offsets",
}


@dataclass(frozen=True)
class _PrecomputedRegularFieldData:
    axes: tuple[np.ndarray, ...]
    times: np.ndarray
    valid_mask: np.ndarray
    support_phi: np.ndarray | None
    quantities: dict[str, QuantitySeriesND]
    metadata: dict[str, Any]


def _matching_field_axes(
    payload: Mapping[str, np.ndarray],
    spatial_dim: int,
    scale: float,
    geometry_axes: tuple[np.ndarray, ...],
    npz_path: Path,
) -> tuple[np.ndarray, ...]:
    field_axes = (
        read_axes(payload, spatial_dim, scale_to_si=scale)
        if f"axis_{spatial_dim - 1}" in payload
        else tuple(np.asarray(axis, dtype=np.float64) for axis in geometry_axes)
    )
    if len(field_axes) != len(geometry_axes):
        raise ValueError(f"Field axes must exactly match geometry axes in {npz_path}")
    for axis_index, (field_axis, geometry_axis) in enumerate(
        zip(field_axes, geometry_axes, strict=True)
    ):
        if not float_arrays_equal_ulps(field_axis, geometry_axis):
            raise ValueError(
                f"Field axis_{axis_index} must exactly match "
                f"geometry axis_{axis_index} in {npz_path}"
            )
    return field_axes


def _read_support(
    payload: Mapping[str, np.ndarray],
    expected_shape: tuple[int, ...],
    scale: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    valid_mask = (
        np.asarray(payload["valid_mask"], dtype=bool)
        if "valid_mask" in payload
        else np.ones(expected_shape, dtype=bool)
    )
    if valid_mask.shape != expected_shape:
        raise ValueError(
            "Field valid_mask shape mismatch: "
            f"expected {expected_shape}, got {valid_mask.shape}"
        )
    support_phi = None
    if "support_phi" in payload:
        support_phi = np.asarray(payload["support_phi"], dtype=np.float64) * scale
        if support_phi.shape != expected_shape:
            raise ValueError(
                "Field support_phi shape mismatch: "
                f"expected {expected_shape}, got {support_phi.shape}"
            )
    return valid_mask, support_phi


def _quantity_data(
    payload: Mapping[str, np.ndarray],
    source: str,
    item: Mapping[str, Any],
    spatial_dim: int,
    times: np.ndarray,
) -> np.ndarray | None:
    data = real_quantity_values(payload, source) * float(item.get("scale_to_si", 1.0))
    if data.ndim == spatial_dim:
        return data
    if data.ndim != spatial_dim + 1:
        return None
    if data.shape[0] != times.size:
        raise ValueError(
            f"Quantity {source} time axis mismatch: "
            f"data has {data.shape[0]} steps, times has {times.size}"
        )
    return data


def _validate_quantity_values(
    name: str, data: np.ndarray, valid_mask: np.ndarray, spatial_dim: int
) -> None:
    support = np.asarray(valid_mask, dtype=bool)
    array = np.asarray(data, dtype=np.float64)
    values = array[support] if array.ndim == int(spatial_dim) else array[:, support]
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError(
            f"Quantity {name} contains non-finite values inside field "
            "valid_mask support"
        )


def _read_quantities(
    payload: Mapping[str, np.ndarray],
    mapping: Mapping[str, Mapping[str, Any]],
    spatial_dim: int,
    times: np.ndarray,
    valid_mask: np.ndarray,
    npz_path: Path,
) -> dict[str, QuantitySeriesND]:
    quantities: dict[str, QuantitySeriesND] = {}
    for target, source, item in quantity_sources(
        payload, mapping, _REGULAR_RESERVED_ARRAYS
    ):
        if source not in payload:
            raise ValueError(
                f"Manifest field component {source!r} is missing from {npz_path}"
            )
        data = _quantity_data(payload, source, item, spatial_dim, times)
        if data is None:
            continue
        _validate_quantity_values(target, data, valid_mask, spatial_dim)
        quantities[target] = QuantitySeriesND(
            name=target,
            unit=str(item.get("unit", infer_unit(target))),
            times=times,
            data=data,
            metadata=quantity_metadata(source, item),
        )
    return quantities


def _load_regular_field(
    npz_path: Path,
    spatial_dim: int,
    scale: float,
    geometry_axes: tuple[np.ndarray, ...],
    mapping: Mapping[str, Mapping[str, Any]],
) -> _PrecomputedRegularFieldData:
    with np.load(npz_path, allow_pickle=False) as payload:
        field_axes = _matching_field_axes(
            payload,
            spatial_dim,
            scale,
            geometry_axes,
            npz_path,
        )
        expected_shape = tuple(len(axis) for axis in field_axes)
        valid_mask, support_phi = _read_support(payload, expected_shape, scale)
        times = read_times(payload)
        metadata = read_metadata(payload)
        quantities = _read_quantities(
            payload,
            mapping,
            spatial_dim,
            times,
            valid_mask,
            npz_path,
        )
    return _PrecomputedRegularFieldData(
        field_axes,
        times,
        valid_mask,
        support_phi,
        quantities,
        metadata,
    )


def _field_metadata(
    data: _PrecomputedRegularFieldData,
    npz_path: Path,
    scale: float,
    mapping: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    default_support_kind = (
        "provider_support_phi" if data.support_phi is not None else ""
    )
    return {
        "npz_path": str(npz_path),
        "provider_kind": "precomputed_npz",
        "coordinate_scale_to_si": float(scale),
        "manifest_quantity_mapping": mapping,
        "field_support_phi_kind": str(
            data.metadata.get("field_support_phi_kind", default_support_kind)
        ),
        **data.metadata,
    }


def build_precomputed_field(
    cfg: Mapping[str, Any],
    spatial_dim: int,
    coordinate_system: str,
    axes: tuple[np.ndarray, ...],
) -> FieldProviderND:
    npz_path = resolve_path(cfg)
    scale = coordinate_scale(cfg)
    mapping = quantity_mapping(cfg)
    data = _load_regular_field(npz_path, spatial_dim, scale, axes, mapping)
    if not data.quantities:
        raise ValueError(f"No field quantities found in {npz_path}")
    time_mode = (
        "transient"
        if any(
            np.asarray(quantity.data).ndim == spatial_dim + 1 and data.times.size > 1
            for quantity in data.quantities.values()
        )
        else "steady"
    )
    field = RegularFieldND(
        spatial_dim=int(spatial_dim),
        coordinate_system=str(coordinate_system),
        axis_names=axis_names_for_coordinate_system(coordinate_system, spatial_dim),
        axes=data.axes,
        quantities=data.quantities,
        valid_mask=data.valid_mask,
        support_phi=data.support_phi,
        time_mode=time_mode,
        metadata=_field_metadata(data, npz_path, scale, mapping),
    )
    return FieldProviderND(
        field=field,
        kind=str(data.metadata.get("provider_kind", "precomputed_npz")),
    )
