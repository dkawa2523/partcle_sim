"""Validate the external COMSOL exporter handoff before transformation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from particle_tracer_unified.integrity import sha256_file as sha256

from .profiles import BuildProfile

_RAW_EXPORT_TEXT_FIELDS = (
    "model_name",
    "study",
    "dataset",
    "solution",
    "comsol_version",
    "mesh_tag",
    "parameter_name",
    "parameter_value",
    "geometry_model_unit",
)

_RAW_EXPORT_ARTIFACTS = (
    ("mesh.mphtxt", "mesh_sha256"),
    ("field_samples.csv", "field_samples_sha256"),
)

# A mesh-native export adds one table evaluated at the COMSOL mesh vertices.
# It is optional so existing grid-only exports stay valid, but it is declared
# in the manifest rather than discovered on disk: the builder must never pick
# a different physics representation from file presence alone.
NODE_SAMPLES_FILENAME = "field_samples_nodes.csv"
NODE_SAMPLES_DIGEST_KEY = "field_node_samples_sha256"


def _sha256_text(value: Any, *, context: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 hex digest")
    return text


def _canonical_export_text(payload: Mapping[str, Any]) -> dict[str, str]:
    if payload.get("source_kind") != "comsol_java_api_external_export":
        raise ValueError(
            "raw export manifest source_kind must be comsol_java_api_external_export"
        )
    normalized: dict[str, str] = {}
    for key in _RAW_EXPORT_TEXT_FIELDS:
        value = payload.get(key)
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(
                f"raw export manifest {key} must be a non-empty canonical string"
            )
        normalized[key] = value
    return normalized


def _solution_metadata(payload: Mapping[str, Any]) -> dict[str, int | float]:
    solution_number = payload.get("solution_number")
    if (
        isinstance(solution_number, bool)
        or not isinstance(solution_number, int)
        or solution_number <= 0
    ):
        raise ValueError(
            "raw export manifest solution_number must be a positive integer"
        )

    scale = payload.get("geometry_scale_m_per_model_unit")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not np.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        raise ValueError(
            "raw export manifest geometry_scale_m_per_model_unit must be "
            "positive and finite"
        )
    if payload.get("solver_coordinate_unit") != "m":
        raise ValueError("raw export manifest solver_coordinate_unit must be exactly m")
    return {
        "solution_number": solution_number,
        "geometry_scale_m_per_model_unit": float(scale),
    }


def _vacuum_domain_ids(payload: Mapping[str, Any]) -> tuple[int, ...]:
    raw_domains = payload.get("vacuum_domain_ids")
    if not isinstance(raw_domains, list):
        raise ValueError(
            "raw export manifest vacuum_domain_ids must be a non-empty integer list"
        )
    domains = [
        int(value)
        for value in raw_domains
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    ]
    if len(domains) != len(raw_domains):
        raise ValueError(
            "raw export manifest vacuum_domain_ids must contain positive integers"
        )
    if not domains or len(set(domains)) != len(domains):
        raise ValueError(
            "raw export manifest vacuum_domain_ids must be non-empty and unique"
        )
    return tuple(domains)


def _export_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = _canonical_export_text(payload)
    normalized.update(_solution_metadata(payload))
    normalized["vacuum_domain_ids"] = _vacuum_domain_ids(payload)
    return normalized


def _profile_expression_units(profile: BuildProfile) -> dict[str, str]:
    return {
        "ux": "m/s",
        "uy": "m/s",
        "E_x": "V/m",
        "E_y": "V/m",
        **{source: unit for source, (_semantic, unit) in profile.scalar_fields.items()},
    }


def _expression_mappings(
    payload: Mapping[str, Any],
) -> tuple[Mapping[Any, Any], Mapping[Any, Any]]:
    expressions = payload.get("expression_mapping")
    units = payload.get("expression_units")
    if not isinstance(expressions, Mapping) or not isinstance(units, Mapping):
        raise ValueError(
            "raw export manifest requires expression_mapping and "
            "expression_units mappings"
        )
    return expressions, units


def _validate_profile_quantities(
    expressions: Mapping[Any, Any],
    units: Mapping[Any, Any],
    profile: BuildProfile,
    expected_units: Mapping[str, str],
) -> None:
    unknown = sorted(set(expressions) - set(expected_units))
    if unknown:
        raise ValueError(
            "raw export manifest contains quantities not declared by the selected "
            f"profile: {unknown}"
        )
    for name in sorted(set(profile.required_sample_columns)):
        if not isinstance(expressions.get(name), str) or not expressions[name].strip():
            raise ValueError(
                "raw export manifest expression_mapping is missing required "
                f"quantity {name!r}"
            )
        if not isinstance(units.get(name), str) or not units[name].strip():
            raise ValueError(
                "raw export manifest expression_units is missing required "
                f"quantity {name!r}"
            )


def _validate_expression_units(
    expressions: Mapping[Any, Any],
    units: Mapping[Any, Any],
    expected_units: Mapping[str, str],
) -> None:
    if set(expressions) != set(units):
        raise ValueError(
            "raw export expression_mapping and expression_units must have "
            "identical keys"
        )
    wrong_units = {
        str(name): {"expected": expected_units[str(name)], "actual": str(units[name])}
        for name in expressions
        if str(units[name]) != expected_units[str(name)]
    }
    if wrong_units:
        raise ValueError(
            "raw export expression units do not match the selected profile: "
            f"{wrong_units}"
        )


def _validated_expression_contract(
    payload: Mapping[str, Any], profile: BuildProfile
) -> tuple[dict[str, str], dict[str, str]]:
    expressions, units = _expression_mappings(payload)
    expected_units = _profile_expression_units(profile)
    _validate_profile_quantities(expressions, units, profile, expected_units)
    _validate_expression_units(expressions, units, expected_units)
    return (
        {str(key): str(value) for key, value in expressions.items()},
        {str(key): str(value) for key, value in units.items()},
    )


def _validated_artifact_hashes(
    raw_dir: Path, payload: Mapping[str, Any]
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for filename, digest_key in _RAW_EXPORT_ARTIFACTS:
        expected = _sha256_text(
            payload.get(digest_key), context=f"raw export manifest {digest_key}"
        )
        artifact_path = raw_dir / filename
        if not artifact_path.is_file():
            raise ValueError(f"raw export artifact is missing: {artifact_path}")
        actual = sha256(artifact_path)
        if actual != expected:
            raise ValueError(
                f"raw export artifact hash mismatch for {filename}: "
                f"expected {expected}, got {actual}"
            )
        hashes[digest_key] = expected
    if payload.get(NODE_SAMPLES_DIGEST_KEY) is not None:
        expected = _sha256_text(
            payload.get(NODE_SAMPLES_DIGEST_KEY),
            context=f"raw export manifest {NODE_SAMPLES_DIGEST_KEY}",
        )
        artifact_path = raw_dir / NODE_SAMPLES_FILENAME
        if not artifact_path.is_file():
            raise ValueError(f"raw export artifact is missing: {artifact_path}")
        actual = sha256(artifact_path)
        if actual != expected:
            raise ValueError(
                "raw export artifact hash mismatch for "
                f"{NODE_SAMPLES_FILENAME}: expected {expected}, got {actual}"
            )
        hashes[NODE_SAMPLES_DIGEST_KEY] = expected
    return hashes


def _validate_sample_quantities(
    raw_dir: Path,
    profile: BuildProfile,
    expressions: Mapping[str, str],
) -> None:
    sample_columns = set(pd.read_csv(raw_dir / "field_samples.csv", nrows=0).columns)
    sampled_quantities = sample_columns - {*profile.sample_axis_columns, "valid_mask"}
    declared_quantities = set(expressions)
    if sampled_quantities != declared_quantities:
        raise ValueError(
            "raw field sample quantities must exactly match export manifest "
            "expression_mapping: "
            f"samples_only={sorted(sampled_quantities - declared_quantities)}, "
            f"manifest_only={sorted(declared_quantities - sampled_quantities)}"
        )


def _artifact_contract(
    raw_dir: Path,
    manifest_path: Path,
    payload: Mapping[str, Any],
    profile: BuildProfile,
    expressions: Mapping[str, str],
) -> dict[str, Any]:
    normalized: dict[str, Any] = _validated_artifact_hashes(raw_dir, payload)
    _validate_sample_quantities(raw_dir, profile, expressions)
    normalized["mph_sha256"] = _sha256_text(
        payload.get("mph_sha256"), context="raw export manifest mph_sha256"
    )
    normalized["config_sha256"] = _sha256_text(
        payload.get("config_sha256"), context="raw export manifest config_sha256"
    )
    normalized["manifest_sha256"] = sha256(manifest_path)
    normalized["manifest_size_bytes"] = int(manifest_path.stat().st_size)
    return normalized


def validate_raw_export(
    raw_dir: Path,
    manifest_path: Path,
    payload: Mapping[str, Any],
    *,
    profile: BuildProfile,
) -> dict[str, Any]:
    """Validate the Java exporter handoff before any case transformation."""

    normalized = _export_metadata(payload)
    expressions, units = _validated_expression_contract(payload, profile)
    normalized["expression_mapping"] = expressions
    normalized["expression_units"] = units
    normalized.update(
        _artifact_contract(raw_dir, manifest_path, payload, profile, expressions)
    )
    return normalized


__all__ = (
    "NODE_SAMPLES_DIGEST_KEY",
    "NODE_SAMPLES_FILENAME",
    "validate_raw_export",
)
