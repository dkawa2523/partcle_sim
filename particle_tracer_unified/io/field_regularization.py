from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, NamedTuple, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ..core.datamodel import FieldProviderND, GeometryProviderND
from ..core.field_sampling import VALID_MASK_STATUS_CLEAN, sample_valid_mask_status
from ..core.grid_sampling import sample_grid_scalar

_DONOR_PROBE_FACTORS: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)


def regularize_precomputed_field_to_geometry(field_provider: FieldProviderND, geometry_provider: GeometryProviderND) -> FieldProviderND:
    field = field_provider.field
    geom = geometry_provider.geometry
    core_valid_mask = np.asarray(field.valid_mask, dtype=bool) & np.asarray(geom.valid_mask, dtype=bool)
    band_distance_m = _max_cell_diagonal(field.axes)
    extension_band_mask = _compute_narrow_band_extension(
        core_valid_mask=core_valid_mask,
        sdf=np.asarray(geom.sdf, dtype=np.float64),
        band_distance_m=float(band_distance_m),
    )
    donor_plan = _build_extension_donor_plan(
        axes=field.axes,
        core_valid_mask=core_valid_mask,
        extension_band_mask=extension_band_mask,
        normal_components=geom.normal_components,
        band_distance_m=float(band_distance_m),
    )
    effective_valid_mask = np.asarray(core_valid_mask | extension_band_mask, dtype=bool)
    regularized_quantities = {
        name: replace(
            series,
            data=_fill_extension_values_from_donors(
                np.asarray(series.data, dtype=np.float64),
                spatial_dim=int(field.spatial_dim),
                axes=field.axes,
                donor_plan=donor_plan,
            ),
        )
        for name, series in field.quantities.items()
    }
    aligned_field = replace(
        field,
        quantities=regularized_quantities,
        valid_mask=effective_valid_mask,
        core_valid_mask=core_valid_mask,
        extension_band_mask=extension_band_mask,
        metadata={
            **field.metadata,
            'effective_valid_mask_from_geometry': True,
            'field_regularization_mode': 'geometry_narrow_band_normal_probe',
            'field_regularization_band_distance_m': float(band_distance_m),
            'field_regularization_added_node_count': int(np.count_nonzero(extension_band_mask)),
            'field_regularization_probe_success_count': int(donor_plan.probe_success_count),
            'field_regularization_probe_fallback_count': int(donor_plan.fallback_count),
            'core_valid_node_count': int(np.count_nonzero(core_valid_mask)),
            'effective_valid_node_count': int(np.count_nonzero(effective_valid_mask)),
        },
    )
    return replace(field_provider, field=aligned_field)


def _max_cell_diagonal(axes: Sequence[np.ndarray]) -> float:
    max_steps = []
    for axis in axes:
        arr = np.asarray(axis, dtype=np.float64)
        if arr.ndim != 1 or arr.size < 2:
            raise ValueError('Field axes must be 1D with at least 2 entries')
        max_steps.append(float(np.max(np.diff(arr))))
    return float(np.linalg.norm(np.asarray(max_steps, dtype=np.float64)))


def _compute_narrow_band_extension(*, core_valid_mask: np.ndarray, sdf: np.ndarray, band_distance_m: float) -> np.ndarray:
    core = np.asarray(core_valid_mask, dtype=bool)
    if float(band_distance_m) <= 0.0:
        return np.zeros_like(core, dtype=bool)
    return np.asarray((~core) & (np.abs(np.asarray(sdf, dtype=np.float64)) <= float(band_distance_m)), dtype=bool)


class _DonorPlan(NamedTuple):
    extension_indices: np.ndarray
    donor_positions: np.ndarray
    donor_core_indices: np.ndarray
    probe_success_count: int
    fallback_count: int


def _build_extension_donor_plan(
    *,
    axes: Tuple[np.ndarray, ...],
    core_valid_mask: np.ndarray,
    extension_band_mask: np.ndarray,
    normal_components: Tuple[np.ndarray, ...],
    band_distance_m: float,
) -> _DonorPlan:
    extension_indices = np.argwhere(np.asarray(extension_band_mask, dtype=bool))
    if extension_indices.size == 0:
        return _DonorPlan(
            np.zeros((0, len(axes)), dtype=np.int32),
            np.zeros((0, len(axes)), dtype=np.float64),
            np.zeros((0, len(axes)), dtype=np.int32),
            0,
            0,
        )

    core = np.asarray(core_valid_mask, dtype=bool)
    core_indices = np.argwhere(core)
    if core_indices.size == 0:
        raise ValueError('Cannot regularize precomputed field without any core-valid nodes')

    core_positions = _indices_to_positions(core_indices, axes)
    tree = cKDTree(core_positions)
    donor_positions = np.zeros((extension_indices.shape[0], len(axes)), dtype=np.float64)
    donor_core_indices = np.zeros((extension_indices.shape[0], len(axes)), dtype=np.int32)
    probe_success_count = 0
    fallback_count = 0

    for row_idx, index in enumerate(extension_indices):
        node_position = _index_to_position(index, axes)
        donor_position = _probe_inward_donor_position(
            axes=axes,
            core_valid_mask=core,
            normal_components=normal_components,
            node_index=index,
            node_position=node_position,
            band_distance_m=float(band_distance_m),
        )
        if donor_position is not None:
            donor_positions[row_idx] = donor_position
            donor_core_indices[row_idx] = _nearest_core_index(tree, core_indices, donor_position)
            probe_success_count += 1
            continue
        donor_index = _nearest_core_index(tree, core_indices, node_position)
        donor_positions[row_idx] = _index_to_position(donor_index, axes)
        donor_core_indices[row_idx] = donor_index
        fallback_count += 1

    return _DonorPlan(
        np.asarray(extension_indices, dtype=np.int32),
        np.asarray(donor_positions, dtype=np.float64),
        np.asarray(donor_core_indices, dtype=np.int32),
        int(probe_success_count),
        int(fallback_count),
    )


def _indices_to_positions(indices: np.ndarray, axes: Tuple[np.ndarray, ...]) -> np.ndarray:
    if indices.size == 0:
        return np.zeros((0, len(axes)), dtype=np.float64)
    cols = [np.asarray(axes[dim], dtype=np.float64)[np.asarray(indices[:, dim], dtype=np.int64)] for dim in range(len(axes))]
    return np.column_stack(cols).astype(np.float64, copy=False)


def _index_to_position(index: Iterable[int], axes: Tuple[np.ndarray, ...]) -> np.ndarray:
    idx = np.asarray(tuple(int(v) for v in index), dtype=np.int64)
    return np.asarray([np.asarray(axes[dim], dtype=np.float64)[idx[dim]] for dim in range(len(axes))], dtype=np.float64)


def _within_axis_bounds(position: np.ndarray, axes: Tuple[np.ndarray, ...]) -> bool:
    point = np.asarray(position, dtype=np.float64)
    for dim, axis in enumerate(axes):
        arr = np.asarray(axis, dtype=np.float64)
        if float(point[dim]) < float(arr[0]) - 1.0e-12 or float(point[dim]) > float(arr[-1]) + 1.0e-12:
            return False
    return True


def _probe_inward_donor_position(
    *,
    axes: Tuple[np.ndarray, ...],
    core_valid_mask: np.ndarray,
    normal_components: Tuple[np.ndarray, ...],
    node_index: np.ndarray,
    node_position: np.ndarray,
    band_distance_m: float,
) -> np.ndarray | None:
    normal = np.asarray([np.asarray(comp, dtype=np.float64)[tuple(int(v) for v in node_index)] for comp in normal_components], dtype=np.float64)
    normal_mag = float(np.linalg.norm(normal))
    if normal_mag <= 1.0e-30:
        return None
    inward = -normal / normal_mag
    for factor in _DONOR_PROBE_FACTORS:
        donor_position = np.asarray(node_position + inward * (float(factor) * float(band_distance_m)), dtype=np.float64)
        if not _within_axis_bounds(donor_position, axes):
            continue
        status = int(sample_valid_mask_status(np.asarray(core_valid_mask, dtype=bool), axes, donor_position))
        if status == int(VALID_MASK_STATUS_CLEAN):
            return donor_position
    return None


def _nearest_core_index(tree: cKDTree, core_indices: np.ndarray, position: np.ndarray) -> np.ndarray:
    _distance, query_index = tree.query(np.asarray(position, dtype=np.float64), k=1)
    return np.asarray(core_indices[int(np.atleast_1d(query_index)[0])], dtype=np.int32)


def _fill_extension_values_from_donors(
    data: np.ndarray,
    *,
    spatial_dim: int,
    axes: Tuple[np.ndarray, ...],
    donor_plan: _DonorPlan,
) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if donor_plan.extension_indices.size == 0:
        return arr.copy()
    out = arr.copy()
    if arr.ndim == int(spatial_dim):
        _fill_spatial_extension_values(out, axes=axes, donor_plan=donor_plan)
        return out
    if arr.ndim == int(spatial_dim) + 1:
        for time_index in range(out.shape[0]):
            _fill_spatial_extension_values(out[time_index], axes=axes, donor_plan=donor_plan)
        return out
    return out


def _fill_spatial_extension_values(
    data: np.ndarray,
    *,
    axes: Tuple[np.ndarray, ...],
    donor_plan: _DonorPlan,
) -> None:
    for row_idx, extension_index in enumerate(donor_plan.extension_indices):
        donor_position = donor_plan.donor_positions[row_idx]
        donor_core_index = donor_plan.donor_core_indices[row_idx]
        value = sample_grid_scalar(np.asarray(data, dtype=np.float64), axes, donor_position)
        if not np.isfinite(value):
            value = float(np.asarray(data, dtype=np.float64)[tuple(int(v) for v in donor_core_index)])
        data[tuple(int(v) for v in extension_index)] = float(value)
