"""Resolve case physics and write the canonical manifest/config pair."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from particle_tracer_unified.core.coordinate_systems import (
    axis_names_for_coordinate_system,
)
from particle_tracer_unified.integrity import sha256_file as sha256

from ._contract_inputs import (
    FIELD_STORAGE_MESH_NATIVE,
    FIELD_STORAGE_REGULAR_GRID,
)
from .fields import field_manifest
from .profiles import COMSOL_MANIFEST_NAME, SCHEMA_VERSION, BuildProfile

if TYPE_CHECKING:
    from particle_tracer_unified.force_models import ForceModel


def _artifact_entry(path: Path, root: Path, artifact_format: str) -> dict[str, Any]:
    return {
        "path": Path(path).resolve().relative_to(Path(root).resolve()).as_posix(),
        "format": artifact_format,
        "sha256": sha256(path),
        "size_bytes": int(Path(path).stat().st_size),
    }


def _load_force_inventory_entries(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read force inventory {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid force inventory YAML {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(
            "force inventory YAML root must be a mapping with a forces list"
        )
    unknown = sorted(str(key) for key in payload if str(key) != "forces")
    if unknown:
        raise ValueError(f"force inventory YAML has unknown keys: {unknown}")
    entries = payload.get("forces")
    if not isinstance(entries, list):
        raise ValueError("force inventory YAML forces must be a list")
    result: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"force inventory YAML forces[{index}] must be a mapping")
        result.append(dict(entry))
    return tuple(result)


def resolve_force_inventory(
    *,
    drag_law: str,
    enabled_forces: tuple[str, ...],
    force_inventory_path: Path | None,
    coordinate_system: str,
) -> ForceModel:
    """Resolve CLI shorthands into the manifest's typed force inventory."""

    from particle_tracer_unified.force_models import parse_manifest_force_model
    from particle_tracer_unified.io.physics_compatibility import (
        validate_coordinate_force_compatibility,
    )

    simple_names = tuple(str(name) for name in enabled_forces)
    unsupported = sorted(set(simple_names) - {"electric"})
    if unsupported:
        raise ValueError(
            "--force only supports electric because it requires no model "
            f"coefficients; use --force-inventory for {unsupported}"
        )
    explicit = (
        ()
        if force_inventory_path is None
        else _load_force_inventory_entries(force_inventory_path)
    )
    if any(entry.get("solver_force") == "drag" for entry in explicit):
        raise ValueError(
            "force inventory YAML must not declare drag; use --drag-law as its "
            "single source"
        )
    raw_entries: tuple[dict[str, Any], ...] = (
        {"solver_force": "drag", "enabled": True, "law": drag_law},
        *({"solver_force": name, "enabled": True} for name in simple_names),
        *explicit,
    )
    model = parse_manifest_force_model(raw_entries, spatial_dim=2)
    validate_coordinate_force_compatibility(coordinate_system, model)
    return model


_GAS_REQUIREMENTS = {
    "none": (),
    "stokes": ("dynamic_viscosity_Pas",),
    "stokes_cunningham": (
        "temperature_K",
        "dynamic_viscosity_Pas",
        "density_kgm3",
        "molecular_mass_amu",
    ),
    "schiller_naumann": ("dynamic_viscosity_Pas", "density_kgm3"),
    "epstein": ("temperature_K", "density_kgm3", "molecular_mass_amu"),
}


def _positive_gas_values(values: Mapping[str, float | None]) -> dict[str, float]:
    gas = {name: float(value) for name, value in values.items() if value is not None}
    nonpositive = [
        name for name, value in gas.items() if not np.isfinite(value) or value <= 0.0
    ]
    if nonpositive:
        raise ValueError(f"gas values must be positive and finite: {nonpositive}")
    return gas


def validate_gas(drag_law: str, values: Mapping[str, float | None]) -> dict[str, float]:
    """Validate the gas properties actually required by one drag law."""

    if drag_law not in _GAS_REQUIREMENTS:
        raise ValueError(f"unsupported drag law: {drag_law!r}")
    missing = [name for name in _GAS_REQUIREMENTS[drag_law] if values.get(name) is None]
    if missing:
        raise ValueError(
            f"drag law {drag_law!r} requires explicit gas values: {missing}"
        )
    return _positive_gas_values(values)


_FIELD_ARTIFACT_FORMATS = {
    FIELD_STORAGE_REGULAR_GRID: "precomputed_npz",
    FIELD_STORAGE_MESH_NATIVE: "precomputed_triangle_mesh_npz",
}


def write_case_contract(
    *,
    out_dir: Path,
    geometry_npz: Path,
    field_npz: Path,
    particles_csv: Path,
    boundaries_csv: Path,
    coordinate_system: str,
    field_storage: str = FIELD_STORAGE_REGULAR_GRID,
    profile: BuildProfile,
    model_provenance: Mapping[str, str],
    force_inventory: ForceModel,
    gas: Mapping[str, float],
    dt_s: float,
    t_end_s: float,
    output_mode: str,
    trajectory_interval_steps: int | None,
    source_metadata: Mapping[str, Any],
) -> None:
    """Write and immediately parse the canonical manifest and run config."""

    from particle_tracer_unified.configuration import RunConfig
    from particle_tracer_unified.force_models import force_model_to_manifest_inventory
    from particle_tracer_unified.io.comsol_manifest import ComsolCaseManifest

    required_provenance = ("name", "study", "dataset", "solution")
    missing = [
        name
        for name in required_provenance
        if not str(model_provenance.get(name, "")).strip()
    ]
    if missing:
        raise ValueError(f"COMSOL model provenance is missing: {missing}")
    fields, time_support = field_manifest(
        field_npz, coordinate_system=coordinate_system, profile=profile
    )
    field_artifact_format = _FIELD_ARTIFACT_FORMATS.get(str(field_storage))
    if field_artifact_format is None:
        raise ValueError(
            f"unknown COMSOL field storage {field_storage!r}; expected one of "
            f"{sorted(_FIELD_ARTIFACT_FORMATS)}"
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": {name: str(model_provenance[name]) for name in required_provenance},
        "coordinates": {
            "coordinate_system": coordinate_system,
            "axis_order": list(axis_names_for_coordinate_system(coordinate_system, 2)),
            "coordinate_scale_m_per_model_unit": 1.0,
        },
        "time": {"interpolation": "linear", "support_s": list(time_support)},
        "artifacts": {
            "release": _artifact_entry(
                particles_csv, out_dir, "canonical_particles_csv"
            ),
            "geometry": _artifact_entry(geometry_npz, out_dir, "precomputed_npz"),
            "field": _artifact_entry(field_npz, out_dir, field_artifact_format),
            "boundaries": _artifact_entry(
                boundaries_csv, out_dir, "canonical_boundaries_csv"
            ),
        },
        "fields": fields,
        "forces": list(force_model_to_manifest_inventory(force_inventory)),
        "metadata": {
            "builder": "particle-tracer comsol build-case",
            "profile": profile.name,
            "field_storage": str(field_storage),
            **dict(source_metadata),
        },
    }
    manifest_path = out_dir / COMSOL_MANIFEST_NAME
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    ComsolCaseManifest.load(manifest_path).validate(strict=True)

    output: dict[str, Any] = {"mode": output_mode}
    if output_mode == "debug":
        output["trajectory_interval_steps"] = int(trajectory_interval_steps or 1)
    config = {
        "schema_version": SCHEMA_VERSION,
        "case": {
            "spatial_dim": 2,
            "coordinate_system": coordinate_system,
            "adapter": "comsol",
        },
        "inputs": {"comsol_manifest": COMSOL_MANIFEST_NAME},
        "physics": {
            "gas": dict(gas),
            "forces": {},
            "seed": 12345,
            # COMSOL's particle tracing has no contact model for point
            # particles: it keeps resolving individual bounces.  A case built
            # to reproduce a COMSOL run therefore does not latch a particle
            # into the sliding state this solver otherwise uses to bound
            # repeated same-wall contact.
            "wall_interaction": {"contact_sliding": False},
        },
        "time": {"dt": float(dt_s), "t_end": float(t_end_s)},
        "output": output,
    }
    typed = RunConfig.from_mapping(config)
    (out_dir / "run_config.yaml").write_text(
        yaml.safe_dump(typed.to_mapping(), sort_keys=False), encoding="utf-8"
    )


__all__ = ("resolve_force_inventory", "validate_gas", "write_case_contract")
