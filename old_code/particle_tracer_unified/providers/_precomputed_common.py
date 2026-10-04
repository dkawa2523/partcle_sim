"""Shared array and mapping validation for precomputed providers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def resolve_path(cfg: Mapping[str, Any], key: str = "npz_path") -> Path:
    value = cfg.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"providers.{key} is required for precomputed_npz providers")
    return Path(str(value)).resolve()


def coordinate_scale(cfg: Mapping[str, Any]) -> float:
    scale = float(cfg.get("coordinate_scale_to_si", 1.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("coordinate_scale_to_si must be positive and finite")
    return scale


def read_axes(
    payload: Mapping[str, np.ndarray],
    spatial_dim: int,
    *,
    scale_to_si: float = 1.0,
) -> tuple[np.ndarray, ...]:
    axes = []
    for index in range(spatial_dim):
        key = f"axis_{index}"
        if key not in payload:
            raise ValueError(f"Missing axis in npz: {key}")
        axis = np.asarray(payload[key], dtype=np.float64) * float(scale_to_si)
        if axis.ndim != 1 or axis.size < 2:
            raise ValueError(f"Axis {key} must be 1D with at least 2 entries")
        if not np.all(np.isfinite(axis)):
            raise ValueError(f"Axis {key} must contain only finite values")
        if not np.all(np.diff(axis) > 0.0):
            raise ValueError(f"Axis {key} must be strictly increasing")
        axes.append(axis)
    return tuple(axes)


def read_times(payload: Mapping[str, np.ndarray]) -> np.ndarray:
    times = (
        np.asarray(payload["times"], dtype=np.float64)
        if "times" in payload
        else np.asarray([0.0], dtype=np.float64)
    )
    if times.ndim != 1 or times.size == 0:
        raise ValueError("Field times must be a non-empty 1D array")
    if not np.all(np.isfinite(times)):
        raise ValueError("Field times must contain only finite values")
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise ValueError("Field times must be strictly increasing")
    return times


def read_metadata(payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    raw: Any = payload.get("metadata_json")
    if raw is None:
        return {}
    if isinstance(raw, np.ndarray):
        if raw.ndim == 0:
            raw = raw.item()
        elif raw.size == 1:
            raw = raw.reshape(()).item()
    if raw is None:
        return {}
    try:
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        data = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def quantity_mapping(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = cfg.get("quantity_mapping", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("quantity_mapping must be a mapping")
    result: dict[str, dict[str, Any]] = {}
    for target, item in raw.items():
        if not isinstance(item, Mapping):
            raise ValueError(f"quantity_mapping.{target} must be a mapping")
        source = str(item.get("source", "")).strip()
        if not source:
            raise ValueError(f"quantity_mapping.{target}.source is required")
        scale = float(item.get("scale_to_si", 1.0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"quantity_mapping.{target}.scale_to_si must be positive and finite"
            )
        result[str(target)] = {
            "source": source,
            "unit": str(item.get("unit", "")),
            "scale_to_si": scale,
            "semantic_quantity": str(item.get("semantic_quantity", "")),
            "component": str(item.get("component", "")),
        }
    return result


def infer_unit(name: str) -> str:
    if name in {
        "ux",
        "uy",
        "uz",
        "ur",
        "vz",
        "u_tau",
        "utau",
        "friction_velocity",
        "friction_velocity_mps",
    }:
        return "m/s"
    if name in {"mu", "dynamic_viscosity", "dynamic_viscosity_Pas"}:
        return "Pa*s"
    if name in {"tauw", "tau_wall", "wall_shear_stress", "tauw_mag"}:
        return "Pa"
    return ""


def quantity_sources(
    payload: Mapping[str, np.ndarray],
    mapping: Mapping[str, Mapping[str, Any]],
    reserved: set[str],
) -> list[tuple[str, str, Mapping[str, Any]]]:
    if mapping:
        return [(target, str(item["source"]), item) for target, item in mapping.items()]
    return [
        (key, key, {})
        for key in payload
        if key not in reserved and not key.startswith("axis_")
    ]


def quantity_metadata(source: str, item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_array": source,
        "semantic_quantity": str(item.get("semantic_quantity", "")),
        "component": str(item.get("component", "")),
        "scale_to_si": float(item.get("scale_to_si", 1.0)),
    }


def real_quantity_values(payload: Mapping[str, np.ndarray], source: str) -> np.ndarray:
    values = np.asarray(payload[source])
    if np.iscomplexobj(values):
        raise ValueError(
            f"Quantity {source} must be real-valued; "
            "complex field data needs an explicit representation"
        )
    return np.asarray(values, dtype=np.float64)
