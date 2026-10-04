"""Resolve native or COMSOL inputs into one typed runtime boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from particle_tracer_unified.configuration import ProviderConfig, RunConfig
from particle_tracer_unified.force_models import ForceModel

from .comsol import load_comsol_runtime_inputs, validate_comsol_runtime_provider
from .comsol_manifest import ComsolCaseManifest
from .runtime_builder_support import (
    LoadedRuntimeInputs,
    RuntimeProviders,
    build_runtime_providers,
    load_runtime_inputs,
    resolve_runtime_input_paths,
)


@dataclass(frozen=True)
class ResolvedAdapterInputs:
    runtime_inputs: LoadedRuntimeInputs
    providers: RuntimeProviders
    force_model: ForceModel
    drag_model: str


def _provider_mapping(config: ProviderConfig) -> dict[str, object]:
    result: dict[str, object] = {
        "kind": config.kind,
        **deepcopy(dict(config.parameters)),
    }
    if config.path is not None:
        result["npz_path"] = config.path
    return result


def _load_manifest(config: RunConfig, config_dir: Path) -> ComsolCaseManifest:
    manifest_value = config.inputs.comsol_manifest
    if manifest_value is None:
        raise ValueError("COMSOL cases require inputs.comsol_manifest")
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = (config_dir / manifest_path).resolve()
    manifest = ComsolCaseManifest.load(manifest_path)
    manifest.validate(strict=True)
    if manifest.coordinate_system != config.case.coordinate_system:
        raise ValueError(
            "case.coordinate_system must match COMSOL manifest "
            "coordinates.coordinate_system: "
            f"{config.case.coordinate_system!r} != {manifest.coordinate_system!r}"
        )
    return manifest


def _native_inputs(config: RunConfig, config_dir: Path) -> ResolvedAdapterInputs:
    inputs = config.inputs
    if inputs.particles is None or inputs.boundaries is None:
        raise ValueError("native solver context requires particles and boundaries")
    if inputs.geometry is None or inputs.field is None:
        raise ValueError("native solver context requires geometry and field providers")

    runtime_inputs = load_runtime_inputs(
        paths=resolve_runtime_input_paths(
            config_dir,
            {
                "particles_csv": inputs.particles,
                "boundaries_csv": inputs.boundaries,
            },
        ),
        spatial_dim=config.case.spatial_dim,
        coordinate_system=config.case.coordinate_system,
    )
    providers = build_runtime_providers(
        config_dir=config_dir,
        providers_cfg={
            "geometry": _provider_mapping(inputs.geometry),
            "field": _provider_mapping(inputs.field),
        },
        spatial_dim=config.case.spatial_dim,
        coordinate_system=config.case.coordinate_system,
    )
    force_model = config.physics.force_model
    if force_model is None:
        raise ValueError("native cases require a typed force model")
    return ResolvedAdapterInputs(
        runtime_inputs=runtime_inputs,
        providers=providers,
        force_model=force_model,
        drag_model=force_model.drag.model,
    )


def _comsol_inputs(config: RunConfig, config_dir: Path) -> ResolvedAdapterInputs:
    manifest = _load_manifest(config, config_dir)
    runtime_inputs = load_comsol_runtime_inputs(
        manifest=manifest,
        spatial_dim=config.case.spatial_dim,
    )
    providers = build_runtime_providers(
        config_dir=config_dir,
        providers_cfg=manifest.provider_config(),
        spatial_dim=config.case.spatial_dim,
        coordinate_system=config.case.coordinate_system,
    )
    validate_comsol_runtime_provider(manifest, providers.field_provider)
    force_model = manifest.force_model
    drag_model = force_model.drag.model
    config.physics.gas.require_for_drag(drag_model)
    return ResolvedAdapterInputs(
        runtime_inputs=runtime_inputs,
        providers=providers,
        force_model=force_model,
        drag_model=drag_model,
    )


def resolve_adapter_inputs(
    config: RunConfig,
    config_dir: Path,
) -> ResolvedAdapterInputs:
    if config.case.adapter == "native":
        return _native_inputs(config, config_dir)
    return _comsol_inputs(config, config_dir)


__all__ = ("ResolvedAdapterInputs", "resolve_adapter_inputs")
