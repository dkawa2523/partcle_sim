"""Assemble canonical physics and run configuration from legacy sections."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from particle_tracer_unified.configuration import RunConfig
from particle_tracer_unified.core.coordinate_systems import normalize_coordinate_system

from .charge import _migrate_charge
from .forces import (
    _legacy_drag_model,
    _migrate_forces,
    _normalized_force_entries,
)
from .legacy import (
    _canonical_choice,
    _canonical_keys,
    _legacy_bool,
    _mapping,
    _merge_without_conflicts,
    _relocated_reference,
)
from .providers import _provider


def _migrate_stochastic(solver: Mapping[str, Any]) -> dict[str, Any] | None:
    raw_present = "stochastic_motion" in solver
    raw = solver.get("stochastic_motion")
    forces = _normalized_force_entries(solver)
    brownian_present = "brownian" in forces
    brownian_raw = forces.get("brownian")
    if not raw_present and not brownian_present:
        return None
    aliases = {
        "active": "enabled",
        "random_seed": "seed",
        "temperature": "temperature_source",
        "update_stride": "stride",
    }
    cfg: dict[str, Any] = {}
    if raw_present:
        standalone = (
            _canonical_keys(
                dict(raw),
                canonical=("enabled", "model", "temperature_source", "seed", "stride"),
                aliases=aliases,
                label="solver.stochastic_motion",
            )
            if isinstance(raw, Mapping)
            else {"enabled": raw}
        )
        standalone.setdefault("enabled", False)
        _merge_without_conflicts(cfg, standalone, label="stochastic configuration")
    if brownian_present:
        brownian = (
            _canonical_keys(
                dict(brownian_raw),
                canonical=("enabled", "model", "temperature_source", "seed", "stride"),
                aliases=aliases,
                label="solver.forces.brownian",
            )
            if isinstance(brownian_raw, Mapping)
            else {"enabled": brownian_raw}
        )
        brownian.setdefault("enabled", True)
        _merge_without_conflicts(cfg, brownian, label="stochastic configuration")
    enabled = _legacy_bool(cfg.get("enabled", False), default=False)
    # The retired implementation defaulted enabled Brownian motion to a stride
    # approximation. Do not silently turn that into every-substep noise.
    stride = int(cfg.get("stride", 10 if enabled else 1))
    if stride != 1:
        raise ValueError(
            "legacy Brownian stride cannot be migrated; "
            "v0.2 updates every accepted substep"
        )
    result: dict[str, Any] = {
        "enabled": enabled,
        "model": _canonical_choice(
            cfg.get("model", "underdamped_langevin"),
            canonical=("underdamped_langevin",),
            aliases={
                "langevin": "underdamped_langevin",
                "underdamped": "underdamped_langevin",
            },
            label="stochastic model",
        ),
        "temperature_source": _canonical_choice(
            cfg.get("temperature_source", "field_T_then_gas"),
            canonical=("field_T_then_gas", "gas"),
            aliases={
                "field_then_gas": "field_T_then_gas",
                "field_temperature_then_gas": "field_T_then_gas",
                "gas_only": "gas",
            },
            label="stochastic temperature source",
        ),
    }
    if cfg.get("seed") is not None:
        result["seed"] = int(cfg["seed"])
    return result


def _gas_config(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("gas", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("legacy gas must be a mapping")
    gas = _canonical_keys(
        dict(raw),
        canonical=(
            "temperature_K",
            "dynamic_viscosity_Pas",
            "density_kgm3",
            "molecular_mass_amu",
        ),
        aliases={
            "temperature": "temperature_K",
            "viscosity": "dynamic_viscosity_Pas",
            "viscosity_pas": "dynamic_viscosity_Pas",
            "density": "density_kgm3",
            "molecular_mass": "molecular_mass_amu",
        },
        label="gas",
    )
    return {name: float(value) for name, value in gas.items() if value is not None}


def _legacy_adapter(legacy: Mapping[str, Any]) -> str:
    mode = str(legacy.get("mode", "")).strip().lower()
    manifest = _mapping(legacy.get("comsol")).get("manifest")
    return "comsol" if mode == "comsol_faithful" or manifest else "native"


def _canonical_config(
    legacy: Mapping[str, Any],
    *,
    source_base: Path,
    destination_base: Path,
    warnings: list[str],
) -> RunConfig:
    run = _mapping(legacy.get("run"))
    solver = _mapping(legacy.get("solver"))
    providers = _mapping(legacy.get("providers"))
    spatial_dim = int(run.get("spatial_dim", 2))
    coordinate_system = normalize_coordinate_system(
        run.get("coordinate_system"), spatial_dim
    )
    adapter = _legacy_adapter(legacy)

    if str(solver.get("integrator", "drag_relaxation")).strip().lower() != "etd2":
        warnings.append(
            "legacy integrator was replaced by the v0.2 fixed ETD2 integrator"
        )
    for obsolete in ("min_tau_p_s", "valid_mask_policy"):
        if obsolete in solver:
            warnings.append(f"solver.{obsolete} was removed")

    inputs: dict[str, Any]
    physics: dict[str, Any] = {
        "gas": _gas_config(legacy),
        "forces": {},
        "seed": int(solver.get("seed", 12345)),
    }
    if adapter == "comsol":
        manifest = _mapping(legacy.get("comsol")).get("manifest")
        if manifest is None or not str(manifest).strip():
            raise ValueError("legacy COMSOL mode requires comsol.manifest")
        inputs = {
            "comsol_manifest": _relocated_reference(
                source_base, destination_base, manifest
            ),
        }
    else:
        inputs = {
            "particles": "particles.csv",
            "boundaries": "boundaries.csv",
            "geometry": _provider(
                providers.get("geometry"),
                source_base=source_base,
                destination_base=destination_base,
                label="geometry",
            ),
            "field": _provider(
                providers.get("field"),
                source_base=source_base,
                destination_base=destination_base,
                label="field",
            ),
        }
        physics["drag"] = {"model": _legacy_drag_model(solver, warnings)}
        physics["forces"] = _migrate_forces(solver)
    charge = _migrate_charge(solver, warnings)
    if charge is not None:
        physics["charge"] = charge
    stochastic = _migrate_stochastic(solver)
    if stochastic is not None:
        physics["stochastic"] = stochastic

    output_raw = _mapping(legacy.get("output"))
    mode = _canonical_choice(
        output_raw.get("mode", output_raw.get("artifact_mode", "standard")),
        canonical=("standard", "debug"),
        aliases={
            "full": "debug",
            "diagnostic": "debug",
            "diagnostics": "debug",
            "normal": "standard",
        },
        label="output mode",
    )
    output: dict[str, Any] = {"mode": mode}
    if mode == "debug":
        trajectory_interval: Any = output_raw.get(
            "save_every", solver.get("save_every", 10)
        )
        output["trajectory_interval_steps"] = int(trajectory_interval)

    mapping = {
        "schema_version": 2,
        "case": {
            "spatial_dim": spatial_dim,
            "coordinate_system": coordinate_system,
            "adapter": adapter,
        },
        "inputs": inputs,
        "physics": physics,
        "time": {
            "dt": float(solver.get("dt", 1.0e-3)),
            "t_end": float(solver.get("t_end", 0.1)),
        },
        "output": output,
    }
    return RunConfig.from_mapping(mapping)
