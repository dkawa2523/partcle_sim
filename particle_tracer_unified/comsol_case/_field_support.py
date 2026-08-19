from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._field_normalization import (
    _RESERVED_ARRAYS,
    _field_axis_alignment_summary,
    _load_npz_payload,
    _normalize_bundle,
)


@dataclass(frozen=True)
class PackedField:
    """Normalized field artifact summary and particle-release support."""

    summary: Mapping[str, Any]
    particle_valid_mask: np.ndarray


@dataclass(frozen=True)
class _FieldSupport:
    bundle_mask: np.ndarray
    finite_mask: np.ndarray
    invalid_claimed: np.ndarray
    field_mask: np.ndarray
    geometry_mask: np.ndarray
    particle_mask: np.ndarray
    support_phi: np.ndarray | None
    source_kind: str


def _quantity_keys(payload: Mapping[str, np.ndarray]) -> list[str]:
    return [str(name) for name in payload if name not in _RESERVED_ARRAYS]


def _finite_support(
    payload: Mapping[str, np.ndarray], expected_shape: tuple[int, int]
) -> np.ndarray:
    support = np.ones(expected_shape, dtype=bool)
    for name in _quantity_keys(payload):
        array = np.asarray(payload[name], dtype=np.float64)
        finite = (
            np.all(np.isfinite(array), axis=0)
            if array.ndim == 3
            else np.isfinite(array)
        )
        support &= finite
    return support


def _masked(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    result = np.asarray(data, dtype=np.float64).copy()
    if result.ndim == 2:
        result[~valid_mask] = 0.0
    else:
        result[:, ~valid_mask] = 0.0
    return result


def _support_quality(support_phi: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
    support_phi = np.asarray(support_phi, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if support_phi.shape != valid_mask.shape:
        raise ValueError(
            f"support_phi shape mismatch: expected {valid_mask.shape}, "
            f"got {support_phi.shape}"
        )
    finite = np.isfinite(support_phi)
    values = support_phi[finite]
    summary: dict[str, Any] = {
        "grid_node_count": int(support_phi.size),
        "finite_node_count": int(np.count_nonzero(finite)),
        "nonfinite_node_count": int(np.count_nonzero(~finite)),
        "positive_node_count": int(np.count_nonzero(values > 0.0)),
        "zero_node_count": int(np.count_nonzero(values == 0.0)),
        "negative_node_count": int(np.count_nonzero(values < 0.0)),
        "valid_mask_inside_node_count": int(np.count_nonzero(valid_mask)),
        "inside_nonpositive_count": int(
            np.count_nonzero(valid_mask & finite & (support_phi <= 0.0))
        ),
        "outside_positive_count": int(
            np.count_nonzero((~valid_mask) & finite & (support_phi > 0.0))
        ),
    }
    if values.size:
        summary.update(
            min=float(np.min(values)),
            max=float(np.max(values)),
            mean=float(np.mean(values)),
        )
    return summary


def _field_support(
    payload: Mapping[str, np.ndarray],
    geometry_inside: np.ndarray,
) -> _FieldSupport:
    bundle_mask = np.asarray(payload["valid_mask"], dtype=bool)
    finite_mask = _finite_support(payload, bundle_mask.shape)
    invalid_claimed = bundle_mask & ~finite_mask
    if np.any(invalid_claimed):
        raise ValueError(
            "field bundle valid_mask marks non-finite field values as valid; "
            f"invalid_claimed_node_count={int(np.count_nonzero(invalid_claimed))}"
        )
    field_mask = bundle_mask & finite_mask
    geometry_mask = np.asarray(geometry_inside, dtype=bool)
    support_phi = (
        np.asarray(payload["support_phi"], dtype=np.float64)
        if "support_phi" in payload
        else None
    )
    return _FieldSupport(
        bundle_mask=bundle_mask,
        finite_mask=finite_mask,
        invalid_claimed=invalid_claimed,
        field_mask=field_mask,
        geometry_mask=geometry_mask,
        particle_mask=field_mask & geometry_mask,
        support_phi=support_phi,
        source_kind=(
            "finite_field_quantities"
            if bool(np.all(bundle_mask))
            else "bundle_valid_mask_and_finite_field_quantities"
        ),
    )


def _field_support_counts(support: _FieldSupport) -> dict[str, int]:
    return {
        "bundle_valid_node_count": int(np.count_nonzero(support.bundle_mask)),
        "finite_field_node_count": int(np.count_nonzero(support.finite_mask)),
        "provider_support_expanded_node_count": int(
            np.count_nonzero(support.finite_mask & ~support.bundle_mask)
        ),
        "provider_support_removed_nonfinite_node_count": int(
            np.count_nonzero(support.invalid_claimed)
        ),
        "provider_support_outside_geometry_node_count": int(
            np.count_nonzero(support.field_mask & ~support.geometry_mask)
        ),
    }


def _field_output(
    payload: Mapping[str, np.ndarray],
    support: _FieldSupport,
    axes_x: np.ndarray,
    axes_y: np.ndarray,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    output = {
        "axis_0": np.asarray(axes_x, dtype=np.float64),
        "axis_1": np.asarray(axes_y, dtype=np.float64),
        "times": np.asarray(payload["times"], dtype=np.float64),
        "valid_mask": support.field_mask,
        "metadata_json": np.asarray(json.dumps(metadata)),
    }
    if support.support_phi is not None:
        output["support_phi"] = support.support_phi
    for name, value in payload.items():
        if name not in _RESERVED_ARRAYS:
            output[name] = _masked(value, support.field_mask)
    return output


def _field_summary(
    *,
    source: Path,
    alignment: Mapping[str, Any],
    output: Mapping[str, Any],
    support: _FieldSupport,
    geometry_sdf: np.ndarray,
) -> dict[str, Any]:
    return {
        "mode": "validated_export_bundle",
        "bundle_path": str(Path(source).resolve()),
        "axis_alignment": alignment,
        "geometry_mask_applied": False,
        "field_ghost_cells": 0,
        "field_valid_mask_source": support.source_kind,
        **_field_support_counts(support),
        "field_valid_node_count": int(np.count_nonzero(support.field_mask)),
        "geometry_valid_node_count": int(np.count_nonzero(support.geometry_mask)),
        "particle_release_valid_node_count": int(
            np.count_nonzero(support.particle_mask)
        ),
        "quantities": sorted(name for name in output if name not in _RESERVED_ARRAYS),
        "support_phi_quality": (
            _support_quality(support.support_phi, support.field_mask)
            if support.support_phi is not None
            else None
        ),
        "geometry_sdf_quality_against_field_valid_mask": _support_quality(
            -np.asarray(geometry_sdf, dtype=np.float64),
            support.field_mask,
        ),
        "physical_boundary_edge_count": None,
        "field_support_is_physical_boundary": False,
    }


def pack_field_bundle(
    source: Path,
    destination: Path,
    *,
    axes_x: np.ndarray,
    axes_y: np.ndarray,
    geometry_inside: np.ndarray,
    geometry_sdf: np.ndarray,
) -> PackedField:
    """Normalize one export bundle and write its solver field artifact."""

    source_payload = _load_npz_payload(source)
    alignment = _field_axis_alignment_summary(source_payload, axes_x, axes_y)
    payload = _normalize_bundle(source_payload, axes_x, axes_y)
    support = _field_support(payload, geometry_inside)
    metadata = {
        "provider_kind": "precomputed_npz",
        "source_kind": "comsol_export_bundle_field",
        "geometry_mask_applied": False,
        "field_ghost_cells": 0,
        "field_valid_mask_source": support.source_kind,
        **_field_support_counts(support),
        "field_support_phi_kind": (
            "provider_support_phi" if support.support_phi is not None else ""
        ),
        "bundle_path": str(Path(source).resolve()),
        "axis_alignment": alignment,
        "has_domain_region_map": False,
    }
    output = _field_output(payload, support, axes_x, axes_y, metadata)
    summary = _field_summary(
        source=source,
        alignment=alignment,
        output=output,
        support=support,
        geometry_sdf=geometry_sdf,
    )
    np.savez_compressed(destination, **output)
    return PackedField(
        summary=summary,
        particle_valid_mask=support.particle_mask,
    )
