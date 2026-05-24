from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from ..core.catalogs import resolve_step_physics
from ..core.coordinate_systems import axis_names_for_coordinate_system
from ..core.datamodel import PreparedRuntime
from ..core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
    VALID_MASK_STATUS_MIXED_STENCIL,
)
from ..io.runtime_builder import build_runtime_from_config, prepare_runtime
from ..solvers.compiled_field_backend import (
    compile_runtime_backend,
    sample_compiled_acceleration_vectors,
    sample_compiled_flow_vectors,
    sample_compiled_gas_properties_vectors,
    sample_compiled_valid_mask_statuses,
)
from ..solvers.forces import ForceRuntimeParameters, force_catalog_summary, force_runtime_parameters_from_catalog
from ..solvers.integrator_common import effective_tau_from_slip_speed
from ..solvers.particle_state import static_arrays_from_particles
from ..solvers.runtime_plan import build_solver_plan
from ..solvers.solver_entrypoints import run_solver_for_dim
from ._common import write_csv


_AMU_KG = 1.66053906660e-27
_STATUS_NAMES = {
    int(VALID_MASK_STATUS_CLEAN): "clean",
    int(VALID_MASK_STATUS_MIXED_STENCIL): "mixed_stencil",
    int(VALID_MASK_STATUS_HARD_INVALID): "hard_invalid",
}
_FORCE_PREFIXES = (
    "drag",
    "electric",
    "thermo",
    "dielectrophoretic",
    "lift",
    "pressure_gradient",
    "virtual_mass",
    "brownian",
    "external",
    "total",
)
_LEGACY_AXES = ("x", "y", "z")


def _runtime_axis_names(prepared: PreparedRuntime) -> tuple[str, ...]:
    return axis_names_for_coordinate_system(prepared.runtime.coordinate_system, prepared.runtime.spatial_dim)


def _axis_aliases(axis: str, legacy_axis: str) -> tuple[str, ...]:
    seen: list[str] = []
    for value in (axis, legacy_axis):
        if value not in seen:
            seen.append(value)
    return tuple(seen)


def _solver_position_value(row: Mapping[str, Any], axis: str, legacy_axis: str) -> float:
    for name in _axis_aliases(axis, legacy_axis):
        if name in row and pd.notna(row[name]):
            return float(row[name])
    return float("nan")


def _solver_velocity_value(row: Mapping[str, Any], axis: str, legacy_axis: str) -> float:
    for name in _axis_aliases(axis, legacy_axis):
        for candidate in (f"v_{name}", f"v{name}"):
            if candidate in row and pd.notna(row[candidate]):
                return float(row[candidate])
    return float("nan")


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("run config must be a mapping")
    return dict(payload)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _force_one_step_config(
    config: Mapping[str, Any],
    *,
    stochastic_policy: str,
    seed: int | None,
) -> tuple[dict[str, Any], list[str]]:
    cfg = deepcopy(dict(config))
    notes: list[str] = []
    solver = cfg.setdefault("solver", {})
    if not isinstance(solver, dict):
        raise ValueError("solver must be a mapping")
    dt = float(solver.get("dt", 1.0e-3))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("solver.dt must be finite and > 0")
    solver["t_end"] = float(dt)
    solver["save_every"] = 1
    if seed is not None:
        solver["seed"] = int(seed)
        source = cfg.setdefault("source", {})
        if isinstance(source, dict):
            preprocess = source.setdefault("preprocess", {})
            if isinstance(preprocess, dict):
                preprocess["seed"] = int(seed)
        stochastic_cfg = solver.get("stochastic_motion", {})
        if isinstance(stochastic_cfg, Mapping):
            stochastic_next = dict(stochastic_cfg)
            stochastic_next["seed"] = int(seed)
            solver["stochastic_motion"] = stochastic_next
    if str(stochastic_policy) == "off":
        stochastic_cfg = _mapping(solver.get("stochastic_motion", {}))
        stochastic_cfg["enabled"] = False
        solver["stochastic_motion"] = stochastic_cfg
        forces = solver.setdefault("forces", {})
        if not isinstance(forces, dict):
            raise ValueError("solver.forces must be a mapping")
        brownian_cfg = _mapping(forces.get("brownian", {}))
        brownian_cfg["enabled"] = False
        forces["brownian"] = brownian_cfg
        notes.append("stochastic_motion and brownian force disabled for deterministic first-step compare")
    output = cfg.setdefault("output", {})
    if isinstance(output, dict):
        output["artifact_mode"] = "minimal"
        output["save_trajectory"] = False
        output["write_wall_events"] = False
        output["write_step_summary"] = False
        output["write_force_contributions"] = False
        output["write_collision_diagnostics"] = False
    return cfg, notes


def _particle_tau_p(diameter: np.ndarray, density: np.ndarray, gas_mu: np.ndarray, min_tau: float) -> np.ndarray:
    mu = np.maximum(np.asarray(gas_mu, dtype=np.float64), 1.0e-30)
    d = np.maximum(np.asarray(diameter, dtype=np.float64), 1.0e-12)
    rho = np.maximum(np.asarray(density, dtype=np.float64), 1.0e-9)
    return np.maximum(float(min_tau), rho * d * d / (18.0 * mu))


def _single_component_runtime(base: ForceRuntimeParameters, name: str) -> ForceRuntimeParameters:
    flags = {
        "thermophoresis_enabled": False,
        "dielectrophoresis_enabled": False,
        "lift_enabled": False,
        "pressure_gradient_enabled": False,
        "virtual_mass_enabled": False,
        "gravity_buoyancy_enabled": False,
    }
    if name in flags:
        flags[name] = True
    aliases = {
        "thermophoresis": "thermophoresis_enabled",
        "dielectrophoresis": "dielectrophoresis_enabled",
        "lift": "lift_enabled",
        "pressure_gradient": "pressure_gradient_enabled",
        "virtual_mass": "virtual_mass_enabled",
    }
    key = aliases.get(name)
    if key is not None:
        flags[key] = True
    return replace(base, **flags)


def _q_over_m(charge: np.ndarray, mass: np.ndarray) -> np.ndarray:
    q = np.asarray(charge, dtype=np.float64)
    m = np.asarray(mass, dtype=np.float64)
    return np.where(np.isfinite(m) & (np.abs(m) > 1.0e-300), q / m, np.nan)


def _sample_component(
    *,
    compiled: Any,
    prepared: PreparedRuntime,
    positions: np.ndarray,
    velocities: np.ndarray,
    time_s: float,
    force_runtime: ForceRuntimeParameters,
    electric_q_over_m: np.ndarray | None,
) -> np.ndarray:
    particles = prepared.runtime.particles
    if particles is None:
        raise ValueError("Simulation requires particles")
    return sample_compiled_acceleration_vectors(
        compiled,
        int(prepared.runtime.spatial_dim),
        float(time_s),
        positions,
        electric_q_over_m=electric_q_over_m,
        force_runtime=force_runtime,
        particle_diameter=particles.diameter,
        particle_density=particles.density,
        particle_mass=particles.mass,
        dep_particle_rel_permittivity=particles.dep_particle_rel_permittivity,
        thermophoretic_coeff=particles.thermophoretic_coeff,
        velocity=velocities,
        gas_density_kgm3=float(prepared.runtime.gas.density_kgm3),
        gas_mu_pas=float(prepared.runtime.gas.dynamic_viscosity_Pas),
        gas_temperature_K=float(prepared.runtime.gas.temperature),
        gas_molecular_mass_kg=float(prepared.runtime.gas.molecular_mass_amu) * _AMU_KG,
    )


def _force_contribution_frame(prepared: PreparedRuntime) -> pd.DataFrame:
    runtime = prepared.runtime
    particles = runtime.particles
    if particles is None:
        raise ValueError("Simulation requires particles")
    dim = int(runtime.spatial_dim)
    axes = _runtime_axis_names(prepared)
    plan = build_solver_plan(prepared, spatial_dim=dim)
    resolved = prepared.source_preprocess.resolved if prepared.source_preprocess is not None else None
    static = static_arrays_from_particles(particles, plan, resolved=resolved)
    base_phys = resolve_step_physics(runtime.physics_catalog, None)
    force_runtime = force_runtime_parameters_from_catalog(runtime.force_catalog)
    force_by_name = runtime.force_catalog.by_name() if runtime.force_catalog is not None else {}
    compiled = compile_runtime_backend(runtime, dim, particles, force_runtime=force_runtime)

    positions = np.asarray(particles.position[:, :dim], dtype=np.float64)
    velocities = np.asarray(particles.velocity[:, :dim], dtype=np.float64)
    time_s = float(np.nanmin(np.asarray(particles.release_time, dtype=np.float64))) if particles.count else 0.0
    status_codes = sample_compiled_valid_mask_statuses(compiled, positions)
    flow = sample_compiled_flow_vectors(compiled, dim, time_s, positions)
    gas_rho, gas_mu, gas_temp = sample_compiled_gas_properties_vectors(
        compiled,
        dim,
        time_s,
        positions,
        fallback_density_kgm3=float(runtime.gas.density_kgm3),
        fallback_mu_pas=float(runtime.gas.dynamic_viscosity_Pas),
        fallback_temperature_K=float(runtime.gas.temperature),
    )
    global_flow_scale = float(base_phys.get("flow_scale", 1.0))
    global_drag_scale = float(base_phys.get("drag_tau_scale", 1.0))
    global_body_scale = float(base_phys.get("body_accel_scale", 1.0))
    target = flow * (global_flow_scale * static.flow_scale)[:, None]
    slip = np.linalg.norm(velocities - target[:, :dim], axis=1)
    tau_stokes = _particle_tau_p(static.diameter_m, static.density_kgm3, gas_mu, float(base_phys.get("min_tau_p_s", 1.0e-6)))
    tau_stokes = np.maximum(float(base_phys.get("min_tau_p_s", 1.0e-6)), tau_stokes * global_drag_scale * np.maximum(static.drag_tau_scale, 1.0e-6))
    tau_eff = np.asarray(
        [
            effective_tau_from_slip_speed(
                float(tau_stokes[i]),
                float(slip[i]),
                float(static.diameter_m[i]),
                float(gas_rho[i]),
                float(gas_mu[i]),
                int(plan.drag_model_mode),
                float(base_phys.get("min_tau_p_s", 1.0e-6)),
                float(static.density_kgm3[i]),
                float(gas_temp[i]),
                float(runtime.gas.molecular_mass_amu) * _AMU_KG,
            )
            for i in range(particles.count)
        ],
        dtype=np.float64,
    )
    drag = (target[:, :dim] - velocities) / np.maximum(tau_eff, 1.0e-300)[:, None]

    zero_runtime = _single_component_runtime(force_runtime, "")
    qom = _q_over_m(particles.charge, particles.mass)
    zeros = np.zeros((particles.count, dim), dtype=np.float64)
    electric_enabled = bool(getattr(force_by_name.get("electric"), "enabled", False))
    electric = (
        _sample_component(
            compiled=compiled,
            prepared=prepared,
            positions=positions,
            velocities=velocities,
            time_s=time_s,
            force_runtime=zero_runtime,
            electric_q_over_m=qom,
        )
        if electric_enabled
        else zeros.copy()
    )
    components = {
        "drag": drag,
        "electric": electric,
        "thermo": (
            _sample_component(
                compiled=compiled,
                prepared=prepared,
                positions=positions,
                velocities=velocities,
                time_s=time_s,
                force_runtime=_single_component_runtime(force_runtime, "thermophoresis"),
                electric_q_over_m=None,
            )
            if bool(force_runtime.thermophoresis_enabled)
            else zeros.copy()
        ),
        "dielectrophoretic": (
            _sample_component(
                compiled=compiled,
                prepared=prepared,
                positions=positions,
                velocities=velocities,
                time_s=time_s,
                force_runtime=_single_component_runtime(force_runtime, "dielectrophoresis"),
                electric_q_over_m=None,
            )
            if bool(force_runtime.dielectrophoresis_enabled)
            else zeros.copy()
        ),
        "lift": (
            _sample_component(
                compiled=compiled,
                prepared=prepared,
                positions=positions,
                velocities=velocities,
                time_s=time_s,
                force_runtime=_single_component_runtime(force_runtime, "lift"),
                electric_q_over_m=None,
            )
            if bool(force_runtime.lift_enabled)
            else zeros.copy()
        ),
        "pressure_gradient": (
            _sample_component(
                compiled=compiled,
                prepared=prepared,
                positions=positions,
                velocities=velocities,
                time_s=time_s,
                force_runtime=_single_component_runtime(force_runtime, "pressure_gradient"),
                electric_q_over_m=None,
            )
            if bool(force_runtime.pressure_gradient_enabled)
            else zeros.copy()
        ),
        "virtual_mass": (
            _sample_component(
                compiled=compiled,
                prepared=prepared,
                positions=positions,
                velocities=velocities,
                time_s=time_s,
                force_runtime=_single_component_runtime(force_runtime, "virtual_mass"),
                electric_q_over_m=None,
            )
            if bool(force_runtime.virtual_mass_enabled)
            else zeros.copy()
        ),
        "brownian": zeros.copy(),
    }
    body = np.asarray(base_phys.get("body_acceleration", np.zeros(dim)), dtype=np.float64)
    if body.size < dim:
        body = np.pad(body, (0, dim - body.size), constant_values=0.0)
    external = np.tile(body[:dim], (particles.count, 1)) * (global_body_scale * static.body_accel_scale)[:, None]
    if bool(force_runtime.gravity_buoyancy_enabled):
        buoyancy = np.where(static.density_kgm3 > 0.0, 1.0 - gas_rho / np.maximum(static.density_kgm3, 1.0e-300), 1.0)
        external = external * buoyancy[:, None]
    components["external"] = external
    total = np.zeros((particles.count, dim), dtype=np.float64)
    for value in components.values():
        total += np.asarray(value, dtype=np.float64)
    components["total"] = total

    rows: list[dict[str, Any]] = []
    for i in range(particles.count):
        row: dict[str, Any] = {
            "particle_id": int(particles.particle_id[i]),
            "source_part_id": int(particles.source_part_id[i]),
            "time_s": time_s,
            "field_status": _STATUS_NAMES.get(int(status_codes[i]), str(int(status_codes[i]))),
            "notes": "brownian acceleration is stochastic and is reported as zero in deterministic first-step compare",
        }
        for axis_index, axis in enumerate(axes):
            row[axis] = float(positions[i, axis_index])
        for prefix in _FORCE_PREFIXES:
            values = components[prefix]
            for axis_index, axis in enumerate(axes):
                row[f"{prefix}_a{axis}"] = float(values[i, axis_index])
        rows.append(row)
    return pd.DataFrame(rows)


def _first_present(row: Mapping[str, Any], names: Sequence[str]) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            value = row[name]
            if not (isinstance(value, str) and not value.strip()):
                return float(value)
    return float("nan")


def _reference_frame(reference: Path | None, axes: tuple[str, ...]) -> pd.DataFrame | None:
    if reference is None:
        return None
    raw = pd.read_csv(reference)
    if "particle_id" not in raw.columns:
        raise ValueError("reference CSV must contain particle_id")
    rows = []
    for _, row in raw.iterrows():
        item: dict[str, Any] = {"particle_id": int(row["particle_id"])}
        for axis_index, axis in enumerate(axes):
            legacy_axis = _LEGACY_AXES[axis_index]
            position_names: list[str] = []
            velocity_names: list[str] = []
            for name in _axis_aliases(axis, legacy_axis):
                position_names.extend((f"{name}1_ref", f"{name}_ref", f"{name}1", name))
                velocity_names.extend(
                    (
                        f"v{name}1_ref",
                        f"v_{name}1_ref",
                        f"v{name}_ref",
                        f"v_{name}_ref",
                        f"v{name}",
                        f"v_{name}",
                    )
                )
            item[f"{axis}1_ref"] = _first_present(row, tuple(position_names))
            item[f"v{axis}1_ref"] = _first_present(
                row,
                tuple(velocity_names),
            )
        rows.append(item)
    return pd.DataFrame(rows)


def _first_step_error_frame(
    prepared: PreparedRuntime,
    final_particles: pd.DataFrame,
    *,
    reference: Path | None,
) -> pd.DataFrame:
    particles = prepared.runtime.particles
    if particles is None:
        raise ValueError("Simulation requires particles")
    dim = int(prepared.runtime.spatial_dim)
    axes = _runtime_axis_names(prepared)
    rows = []
    final_by_id = final_particles.set_index("particle_id", drop=False)
    for i in range(particles.count):
        pid = int(particles.particle_id[i])
        final = final_by_id.loc[pid] if pid in final_by_id.index else {}
        row: dict[str, Any] = {
            "particle_id": pid,
            "source_part_id": int(particles.source_part_id[i]),
            "field_status": "",
            "notes": "",
        }
        for axis_index, axis in enumerate(axes):
            legacy_axis = _LEGACY_AXES[axis_index]
            row[f"{axis}0"] = float(particles.position[i, axis_index])
            row[f"v{axis}0"] = float(particles.velocity[i, axis_index])
            row_mapping = final if isinstance(final, Mapping) else final.to_dict()
            row[f"{axis}1_solver"] = _solver_position_value(row_mapping, axis, legacy_axis)
            row[f"v{axis}1_solver"] = _solver_velocity_value(row_mapping, axis, legacy_axis)
            row[f"{axis}1_ref"] = float("nan")
            row[f"v{axis}1_ref"] = float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    ref = _reference_frame(reference, axes)
    if ref is not None:
        out = out.drop(columns=[col for col in ref.columns if col in out.columns and col != "particle_id"]).merge(
            ref,
            on="particle_id",
            how="left",
        )
    solver_pos = out[[f"{axis}1_solver" for axis in axes]].to_numpy(dtype=np.float64)
    ref_pos = out[[f"{axis}1_ref" for axis in axes]].to_numpy(dtype=np.float64)
    solver_vel = out[[f"v{axis}1_solver" for axis in axes]].to_numpy(dtype=np.float64)
    ref_vel = out[[f"v{axis}1_ref" for axis in axes]].to_numpy(dtype=np.float64)
    valid_pos = np.all(np.isfinite(solver_pos), axis=1) & np.all(np.isfinite(ref_pos), axis=1)
    valid_vel = np.all(np.isfinite(solver_vel), axis=1) & np.all(np.isfinite(ref_vel), axis=1)
    pos_err = np.full(len(out), np.nan, dtype=np.float64)
    vel_err = np.full(len(out), np.nan, dtype=np.float64)
    speed_ratio = np.full(len(out), np.nan, dtype=np.float64)
    pos_err[valid_pos] = np.linalg.norm(solver_pos[valid_pos] - ref_pos[valid_pos], axis=1)
    vel_err[valid_vel] = np.linalg.norm(solver_vel[valid_vel] - ref_vel[valid_vel], axis=1)
    solver_speed = np.linalg.norm(solver_vel, axis=1)
    ref_speed = np.linalg.norm(ref_vel, axis=1)
    speed_valid = valid_vel & (ref_speed > 1.0e-300)
    speed_ratio[speed_valid] = solver_speed[speed_valid] / ref_speed[speed_valid]
    out["position_error_m"] = pos_err
    out["velocity_error_mps"] = vel_err
    out["speed_ratio"] = speed_ratio
    return out


def _finite_summary(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)) if finite.size else None,
        "max": float(np.max(finite)) if finite.size else None,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def run_first_step_compare(
    *,
    config_path: Path,
    output_dir: Path,
    reference: Path | None = None,
    stochastic: str = "off",
    seed: int | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config, notes = _force_one_step_config(_read_yaml(config_path), stochastic_policy=stochastic, seed=seed)
    runtime = build_runtime_from_config(config, config_path.parent)
    prepared = prepare_runtime(runtime, seed=seed)
    dim = int(prepared.runtime.spatial_dim)

    force_frame = _force_contribution_frame(prepared)
    force_path = output_dir / "force_contributions.csv"
    write_csv(force_frame, force_path)

    solver_dir = output_dir / "_solver_first_step"
    run_solver_for_dim(prepared, solver_dir, spatial_dim=dim)
    final_particles = pd.read_csv(solver_dir / "final_particles.csv")
    first_step_frame = _first_step_error_frame(prepared, final_particles, reference=reference)
    first_step_path = output_dir / "first_step_error.csv"
    write_csv(first_step_frame, first_step_path)

    force_summary = force_catalog_summary(prepared.runtime.force_catalog)
    summary = {
        "config": str(config_path),
        "spatial_dim": int(dim),
        "coordinate_system": str(prepared.runtime.coordinate_system),
        "axis_names": list(_runtime_axis_names(prepared)),
        "particle_count": int(len(first_step_frame)),
        "reference_particle_count": int(0 if reference is None else len(pd.read_csv(reference))),
        "compared_particle_count": int(np.count_nonzero(np.isfinite(first_step_frame["position_error_m"].to_numpy(dtype=float)))),
        "stochastic_policy": str(stochastic),
        "stochastic_disabled_for_compare": int(str(stochastic) == "off"),
        "seed": None if seed is None else int(seed),
        "solver_dt_s": float(_mapping(config.get("solver", {})).get("dt", 1.0e-3)),
        "forced_t_end_s": float(_mapping(config.get("solver", {})).get("t_end", 1.0e-3)),
        "enabled_forces": list(force_summary.get("enabled_forces", [])) if isinstance(force_summary, Mapping) else [],
        "force_contribution_rows": int(len(force_frame)),
        "position_error_m": _finite_summary(first_step_frame["position_error_m"].to_numpy(dtype=float)),
        "velocity_error_mps": _finite_summary(first_step_frame["velocity_error_mps"].to_numpy(dtype=float)),
        "artifacts": {
            "first_step_error_csv": str(first_step_path),
            "force_contributions_csv": str(force_path),
            "solver_output_dir": str(solver_dir),
        },
        "notes": notes,
    }
    summary_path = output_dir / "first_step_compare_summary.json"
    summary["artifacts"]["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a deterministic first-step and force-contribution comparison.")
    parser.add_argument("--config", required=True, type=Path, help="particle_tracer_unified run YAML")
    parser.add_argument("--reference", type=Path, default=None, help="Optional first-step reference CSV")
    parser.add_argument("--output-dir", type=Path, default=Path("first_step_compare"))
    parser.add_argument("--stochastic", choices=("off", "from-config"), default="off")
    parser.add_argument("--seed", type=int, default=None, help="Optional deterministic seed override")
    args = parser.parse_args(argv)

    run_first_step_compare(
        config_path=args.config,
        output_dir=args.output_dir,
        reference=args.reference,
        stochastic=args.stochastic,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
