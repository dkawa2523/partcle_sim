from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from particle_tracer_unified.core.datamodel import (
    FieldProviderND,
    GeometryProviderND,
    ParticleTable,
    PartWallTable,
    RegularFieldND,
    TriangleMeshField2D,
)
from particle_tracer_unified.core.numerical_contracts import float_arrays_equal_ulps
from particle_tracer_unified.io.tables import load_boundaries_csv, load_particles_csv
from particle_tracer_unified.providers.precomputed import (
    build_precomputed_field,
    build_precomputed_geometry,
    build_precomputed_triangle_mesh_field,
)
from particle_tracer_unified.providers.synthetic import (
    build_synthetic_field,
    build_synthetic_geometry,
)


@dataclass(frozen=True)
class ResolvedRuntimePaths:
    particles_path: Path
    boundaries_path: Path


@dataclass(frozen=True)
class LoadedRuntimeInputs:
    particles: ParticleTable
    walls: PartWallTable


@dataclass(frozen=True)
class RuntimeProviders:
    geometry_provider: GeometryProviderND | None
    field_provider: FieldProviderND | None


def _resolve_path(base: Path, value: str | None, *, context: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    if value == "":
        raise ValueError(f"{context} must not be empty")
    if value != value.strip():
        raise ValueError(f"{context} must not contain leading or trailing whitespace")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path


def resolve_runtime_input_paths(
    config_dir: Path, paths_cfg: Mapping[str, Any]
) -> ResolvedRuntimePaths:
    particles_path = _resolve_path(
        config_dir, paths_cfg.get("particles_csv"), context="paths.particles_csv"
    )
    boundaries_path = _resolve_path(
        config_dir, paths_cfg.get("boundaries_csv"), context="paths.boundaries_csv"
    )
    if particles_path is None:
        raise ValueError("paths.particles_csv is required")
    if boundaries_path is None:
        raise ValueError("paths.boundaries_csv is required")
    return ResolvedRuntimePaths(
        particles_path=particles_path,
        boundaries_path=boundaries_path,
    )


def load_runtime_inputs(
    *,
    paths: ResolvedRuntimePaths,
    spatial_dim: int,
    coordinate_system: str,
) -> LoadedRuntimeInputs:
    particles = load_particles_csv(
        paths.particles_path,
        spatial_dim=spatial_dim,
        coordinate_system=coordinate_system,
    )
    walls = load_boundaries_csv(paths.boundaries_path)
    return LoadedRuntimeInputs(
        particles=particles,
        walls=walls,
    )


def _resolved_provider_cfg(
    config_dir: Path, provider_cfg: Mapping[str, Any]
) -> dict[str, Any]:
    resolved_cfg = dict(provider_cfg)
    resolved_npz = _resolve_path(
        config_dir,
        resolved_cfg.get("npz_path"),
        context="providers.npz_path",
    )
    if resolved_npz is not None:
        resolved_cfg["npz_path"] = str(resolved_npz)
    return resolved_cfg


def _align_field_provider_to_geometry(
    field_provider: FieldProviderND,
    geometry_provider: GeometryProviderND,
) -> FieldProviderND:
    field = field_provider.field
    geom = geometry_provider.geometry
    if int(field.spatial_dim) != int(geom.spatial_dim):
        raise ValueError("Field spatial_dim must match geometry spatial_dim")
    if isinstance(field, TriangleMeshField2D):
        return field_provider
    if not isinstance(field, RegularFieldND):
        raise TypeError(f"unsupported field data type: {type(field).__name__}")
    spatial_dim = int(field.spatial_dim)
    if len(field.axes) != spatial_dim or len(geom.axes) != spatial_dim:
        raise ValueError(
            "Field and geometry must each provide exactly spatial_dim axes"
        )
    for axis_index, (field_axis, geometry_axis) in enumerate(
        zip(field.axes, geom.axes, strict=True)
    ):
        if not float_arrays_equal_ulps(field_axis, geometry_axis):
            raise ValueError(
                f"Field axis_{axis_index} must exactly match geometry axis_{axis_index}"
            )
    field_valid_mask = np.asarray(field.valid_mask, dtype=bool)
    geometry_valid_mask = np.asarray(geom.valid_mask, dtype=bool)
    core_valid_mask = field_valid_mask & geometry_valid_mask
    support_phi = getattr(field, "support_phi", None)
    if support_phi is not None:
        support_phi_arr = np.asarray(support_phi, dtype=np.float64)
        if support_phi_arr.shape != field_valid_mask.shape:
            raise ValueError(
                "Field support_phi shape mismatch: "
                f"expected {field_valid_mask.shape}, got {support_phi_arr.shape}"
            )
        support_phi = support_phi_arr
    aligned_field = replace(
        field,
        valid_mask=field_valid_mask,
        support_phi=support_phi,
        core_valid_mask=core_valid_mask,
        metadata={
            **field.metadata,
            "field_valid_mask_is_provider_native": True,
            "field_valid_node_count": int(np.count_nonzero(field_valid_mask)),
            "geometry_valid_node_count": int(np.count_nonzero(geometry_valid_mask)),
            "core_valid_node_count": int(np.count_nonzero(core_valid_mask)),
        },
    )
    return replace(field_provider, field=aligned_field)


def _provider_kind(cfg: Mapping[str, Any], *, context: str, allowed: set[str]) -> str:
    if "kind" not in cfg:
        raise ValueError(f"{context}.kind is required")
    value = cfg["kind"]
    if not isinstance(value, str):
        raise ValueError(f"{context}.kind must be a string")
    if value != value.strip():
        raise ValueError(
            f"{context}.kind must not contain leading or trailing whitespace"
        )
    if value not in allowed:
        raise ValueError(
            f"{context}.kind must be one of {sorted(allowed)}, got {value!r}"
        )
    return value


def _build_geometry_provider(
    config_dir: Path,
    geom_cfg: Mapping[str, Any],
    *,
    spatial_dim: int,
    coordinate_system: str,
) -> GeometryProviderND:
    geom_kind = _provider_kind(
        geom_cfg, context="providers.geometry", allowed={"box", "precomputed_npz"}
    )
    resolved_cfg = _resolved_provider_cfg(config_dir, geom_cfg)
    if geom_kind == "precomputed_npz":
        return build_precomputed_geometry(
            resolved_cfg, spatial_dim=spatial_dim, coordinate_system=coordinate_system
        )
    return build_synthetic_geometry(
        resolved_cfg, spatial_dim=spatial_dim, coordinate_system=coordinate_system
    )


def _build_field_provider(
    config_dir: Path,
    field_cfg: Mapping[str, Any],
    geometry_provider: GeometryProviderND,
    *,
    spatial_dim: int,
    coordinate_system: str,
) -> FieldProviderND:
    field_kind = _provider_kind(
        field_cfg,
        context="providers.field",
        allowed={"linear_shear", "precomputed_npz", "precomputed_triangle_mesh_npz"},
    )
    resolved_cfg = _resolved_provider_cfg(config_dir, field_cfg)
    if field_kind == "precomputed_npz":
        return build_precomputed_field(
            resolved_cfg,
            spatial_dim=spatial_dim,
            coordinate_system=coordinate_system,
            axes=geometry_provider.geometry.axes,
        )
    if field_kind == "precomputed_triangle_mesh_npz":
        return build_precomputed_triangle_mesh_field(
            resolved_cfg,
            spatial_dim=spatial_dim,
            coordinate_system=coordinate_system,
        )
    return build_synthetic_field(
        resolved_cfg,
        spatial_dim=spatial_dim,
        coordinate_system=coordinate_system,
        axes=geometry_provider.geometry.axes,
    )


def build_runtime_providers(
    *,
    config_dir: Path,
    providers_cfg: Mapping[str, Any],
    spatial_dim: int,
    coordinate_system: str,
) -> RuntimeProviders:
    unknown = sorted(set(providers_cfg) - {"geometry", "field"})
    if unknown:
        raise ValueError(f"providers has unknown entries: {unknown}")

    def provider_config(name: str) -> Mapping[str, Any] | None:
        if name not in providers_cfg:
            return None
        value = providers_cfg[name]
        if not isinstance(value, Mapping):
            raise ValueError(f"providers.{name} must be a mapping")
        return value

    geom_cfg = provider_config("geometry")
    field_cfg = provider_config("field")

    geometry_provider = (
        _build_geometry_provider(
            config_dir,
            geom_cfg,
            spatial_dim=spatial_dim,
            coordinate_system=coordinate_system,
        )
        if geom_cfg is not None
        else None
    )
    field_provider = None
    if field_cfg is not None:
        if geometry_provider is None:
            raise ValueError(
                "providers.field requires providers.geometry so axes are available"
            )
        field_provider = _build_field_provider(
            config_dir,
            field_cfg,
            geometry_provider,
            spatial_dim=spatial_dim,
            coordinate_system=coordinate_system,
        )
    if (
        geometry_provider is not None
        and bool(
            geometry_provider.geometry.metadata.get("requires_field_bundle", False)
        )
        and field_provider is None
    ):
        raise ValueError(
            "COMSOL geometry requires providers.field from a validated export bundle"
        )
    if geometry_provider is not None and field_provider is not None:
        field_provider = _align_field_provider_to_geometry(
            field_provider, geometry_provider
        )
    return RuntimeProviders(
        geometry_provider=geometry_provider, field_provider=field_provider
    )


__all__ = (
    "LoadedRuntimeInputs",
    "ResolvedRuntimePaths",
    "RuntimeProviders",
    "build_runtime_providers",
    "load_runtime_inputs",
    "resolve_runtime_input_paths",
)
