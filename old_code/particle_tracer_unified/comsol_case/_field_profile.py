from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from particle_tracer_unified.core.coordinate_systems import (
    axis_names_for_coordinate_system,
)

from .profiles import BuildProfile


@dataclass(frozen=True)
class _ProfileSamples:
    frame: pd.DataFrame
    axis_0_name: str
    axis_1_name: str
    raw_axis_0: np.ndarray
    raw_axis_1: np.ndarray
    coordinate_scale: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.raw_axis_0.size, self.raw_axis_1.size)

    def grid(self, name: str) -> np.ndarray:
        work = self.frame[[self.axis_0_name, self.axis_1_name]].copy()
        work[name] = pd.to_numeric(self.frame[name], errors="coerce")
        return (
            work.pivot(
                index=self.axis_0_name,
                columns=self.axis_1_name,
                values=name,
            )
            .reindex(index=self.raw_axis_0, columns=self.raw_axis_1)
            .to_numpy(dtype=np.float64)
        )


def _profile_samples(
    samples_csv: Path,
    profile: BuildProfile,
    coordinate_scale_m_per_model_unit: float,
) -> _ProfileSamples:
    frame = pd.read_csv(samples_csv)
    axis_0_name, axis_1_name = profile.sample_axis_columns
    missing = sorted(
        {axis_0_name, axis_1_name, *profile.required_sample_columns}
        - set(frame.columns)
    )
    if missing:
        raise ValueError(f"{profile.name} field samples are missing columns: {missing}")
    if frame.duplicated([axis_0_name, axis_1_name]).any():
        raise ValueError(
            f"{profile.name} field samples contain duplicate coordinate pairs"
        )
    scale = float(coordinate_scale_m_per_model_unit)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "coordinate_scale_m_per_model_unit must be positive and finite"
        )
    raw_axis_0 = np.sort(
        np.unique(
            np.asarray(
                pd.to_numeric(frame[axis_0_name], errors="raise"),
                dtype=np.float64,
            )
        )
    )
    raw_axis_1 = np.sort(
        np.unique(
            np.asarray(
                pd.to_numeric(frame[axis_1_name], errors="raise"),
                dtype=np.float64,
            )
        )
    )
    if raw_axis_0.size < 2 or raw_axis_1.size < 2:
        raise ValueError(
            "COMSOL field sample axes must each contain at least two points"
        )
    if len(frame) != int(raw_axis_0.size * raw_axis_1.size):
        raise ValueError("COMSOL field samples must form a complete tensor grid")
    return _ProfileSamples(
        frame=frame,
        axis_0_name=axis_0_name,
        axis_1_name=axis_1_name,
        raw_axis_0=raw_axis_0,
        raw_axis_1=raw_axis_1,
        coordinate_scale=scale,
    )


def _profile_valid_mask(
    samples: _ProfileSamples,
    profile: BuildProfile,
) -> np.ndarray:
    if "valid_mask" in samples.frame:
        valid_mask = samples.grid("valid_mask") > 0.5
    else:
        valid_mask = np.ones(samples.shape, dtype=bool)
        for name in profile.required_sample_columns:
            valid_mask &= np.isfinite(samples.grid(name))
    if not np.any(valid_mask):
        raise ValueError("COMSOL field samples contain no valid support")
    return valid_mask


def _profile_quantities(
    samples: _ProfileSamples,
    profile: BuildProfile,
    valid_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    payload: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}
    reserved = {samples.axis_0_name, samples.axis_1_name, "valid_mask"}
    for name in samples.frame.columns:
        if name in reserved:
            continue
        values = samples.grid(str(name)).copy()
        if np.any(~np.isfinite(values[valid_mask])):
            if name in profile.required_sample_columns:
                raise ValueError(
                    f"required COMSOL field {name!r} is non-finite on valid support"
                )
            skipped[str(name)] = "nonfinite_on_valid_support"
            continue
        values[~valid_mask] = np.nan
        payload[str(name)] = values
    return payload, skipped


def _profile_payload(
    samples: _ProfileSamples,
    profile: BuildProfile,
    samples_csv: Path,
) -> dict[str, np.ndarray]:
    valid_mask = _profile_valid_mask(samples, profile)
    quantities, skipped = _profile_quantities(samples, profile, valid_mask)
    return {
        "axis_0": samples.raw_axis_0 * samples.coordinate_scale,
        "axis_1": samples.raw_axis_1 * samples.coordinate_scale,
        "times": np.asarray([0.0], dtype=np.float64),
        "valid_mask": valid_mask,
        **quantities,
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "profile": profile.name,
                    "source_samples": str(Path(samples_csv).resolve()),
                    "raw_axis_columns": list(profile.sample_axis_columns),
                    "raw_coordinate_scale_m_per_model_unit": samples.coordinate_scale,
                    "artifact_coordinate_unit": "m",
                    "skipped_columns": skipped,
                },
                sort_keys=True,
            )
        ),
    }


def build_profile_field_bundle(
    samples_csv: Path,
    destination: Path,
    *,
    profile: BuildProfile,
    coordinate_scale_m_per_model_unit: float,
) -> Path:
    """Convert a profile-specific COMSOL point table to one SI grid bundle."""

    samples = _profile_samples(
        samples_csv,
        profile,
        coordinate_scale_m_per_model_unit,
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = _profile_payload(samples, profile, samples_csv)
    np.savez_compressed(destination, **payload)
    return destination


def _field_inventory(field_npz: Path) -> tuple[set[str], np.ndarray]:
    with np.load(field_npz, allow_pickle=False) as payload:
        names = set(payload.files)
        times = np.asarray(payload.get("times", [0.0]), dtype=np.float64)
    if times.size == 0 or np.any(~np.isfinite(times)) or np.any(np.diff(times) < 0.0):
        raise ValueError("field bundle times must be finite and non-decreasing")
    if not {"ux", "uy"}.issubset(names):
        raise ValueError("field bundle must contain ux and uy")
    return names, times


def _scalar_manifest_field(
    *,
    profile: BuildProfile,
    source: str,
    semantic: str,
    unit: str,
) -> tuple[str, dict[str, Any]]:
    from particle_tracer_unified.io.comsol_manifest import (
        BUILTIN_FIELD_SEMANTICS,
        classify_field_semantic,
    )

    field_name = source if semantic == "scalar" else semantic
    if semantic == "scalar":
        if (
            classify_field_semantic(field_name) != "scalar"
            or field_name in BUILTIN_FIELD_SEMANTICS
        ):
            raise ValueError(
                f"profile {profile.name!r} generic scalar source {source!r} "
                "must be a non-reserved semantic identifier"
            )
    elif (
        semantic not in BUILTIN_FIELD_SEMANTICS
        or classify_field_semantic(semantic) != "scalar"
    ):
        raise ValueError(
            f"profile {profile.name!r} scalar mapping for {source!r} has "
            f"invalid built-in semantic {semantic!r}"
        )
    return field_name, {
        "artifact": "field",
        "components": {"value": source},
        "unit": unit,
        "scale_to_si": 1.0,
    }


def field_manifest(
    field_npz: Path, *, coordinate_system: str, profile: BuildProfile
) -> tuple[dict[str, Any], tuple[float, float]]:
    """Map artifact arrays to the manifest's physical semantic quantities."""

    axes = axis_names_for_coordinate_system(coordinate_system, 2)
    names, times = _field_inventory(field_npz)
    fields: dict[str, Any] = {
        "velocity": {
            "artifact": "field",
            "components": dict(zip(axes, ("ux", "uy"), strict=True)),
            "unit": "m/s",
            "scale_to_si": 1.0,
        }
    }
    if {"E_x", "E_y"}.issubset(names):
        fields["electric_field"] = {
            "artifact": "field",
            "components": dict(zip(axes, ("E_x", "E_y"), strict=True)),
            "unit": "V/m",
            "scale_to_si": 1.0,
        }
    for source, (semantic, unit) in profile.scalar_fields.items():
        if source not in names:
            continue
        field_name, definition = _scalar_manifest_field(
            profile=profile,
            source=source,
            semantic=semantic,
            unit=unit,
        )
        fields[field_name] = definition
    return fields, (float(times[0]), float(times[-1]))
