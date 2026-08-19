"""Reproducibility provenance collected once at the case-input boundary."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

from particle_tracer_unified.configuration import ProviderConfig, RunConfig
from particle_tracer_unified.integrity import sha256_file

from .comsol_manifest import ComsolCaseManifest


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    return (
        candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    )


def _file_provenance(path: Path, *, configured_path: str) -> dict[str, Any]:
    return {
        "configured_path": str(configured_path),
        "resolved_path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _provider_file(
    result: dict[str, Any],
    *,
    name: str,
    provider: ProviderConfig | None,
    base: Path,
) -> None:
    if provider is None or provider.path is None:
        return
    path = _resolve(base, provider.path)
    result[name] = _file_provenance(path, configured_path=provider.path)


def _canonical_config_sha256(config: RunConfig) -> str:
    canonical = json.dumps(
        config.to_mapping(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def collect_case_provenance(
    config: RunConfig, config_path: str | Path
) -> dict[str, Any]:
    """Describe every external file that selected the resolved case.

    Native NPZ/CSV inputs are hashed with a streaming read.  A COMSOL manifest
    has already verified its artifacts while the solver context was built, so
    this function records those declared hashes instead of rereading large
    exported arrays.
    """

    source = Path(config_path).resolve()
    base = source.parent
    config_record = _file_provenance(source, configured_path=str(config_path))
    config_record["canonical_sha256"] = _canonical_config_sha256(config)
    result: dict[str, Any] = {"config": config_record, "inputs": {}}

    inputs: dict[str, Any] = result["inputs"]
    if config.case.adapter == "native":
        for name, value in (
            ("particles", config.inputs.particles),
            ("boundaries", config.inputs.boundaries),
        ):
            if value is not None:
                inputs[name] = _file_provenance(
                    _resolve(base, value),
                    configured_path=value,
                )
        _provider_file(
            inputs, name="geometry", provider=config.inputs.geometry, base=base
        )
        _provider_file(inputs, name="field", provider=config.inputs.field, base=base)
        return result

    manifest_value = config.inputs.comsol_manifest
    if manifest_value is None:  # guarded by the canonical configuration parser
        raise ValueError("COMSOL cases require inputs.comsol_manifest")
    manifest_path = _resolve(base, manifest_value)
    manifest = ComsolCaseManifest.load(manifest_path)
    manifest_record = _file_provenance(
        manifest_path,
        configured_path=manifest_value,
    )
    manifest_record.update(
        {
            "schema_version": int(manifest.schema_version),
            "model": dict(manifest.model),
            "geometry": {
                "source": manifest.geometry_source,
                "vacuum_domain_ids": list(manifest.vacuum_domain_ids),
            },
            "solution_number": manifest.source_solution_number,
            "artifacts": {
                str(name): {
                    "configured_path": str(artifact.path),
                    "size_bytes": (
                        None
                        if artifact.size_bytes is None
                        else int(artifact.size_bytes)
                    ),
                    "sha256": str(artifact.sha256),
                }
                for name, artifact in sorted(manifest.artifacts.items())
            },
        }
    )
    result["manifest"] = manifest_record
    return result


__all__ = ("collect_case_provenance",)
