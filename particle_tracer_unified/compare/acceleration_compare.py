from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..io.runtime_builder import build_prepared_runtime_from_yaml
from ..solvers.compiled_field_backend import compile_runtime_backend, sample_compiled_acceleration_vector
from ..solvers.forces import force_runtime_parameters_from_catalog
from ._common import (
    component_labels,
    finite_float,
    merge_with_reference,
    read_csv,
    row_particle_id,
    row_point_id,
    row_position,
    row_time,
    row_velocity,
    write_csv,
)


def _particle_defaults(runtime) -> dict[int, dict[str, float]]:
    particles = runtime.particles
    if particles is None:
        return {}
    defaults: dict[int, dict[str, float]] = {}
    for idx, pid in enumerate(np.asarray(particles.particle_id, dtype=np.int64)):
        defaults[int(pid)] = {
            "diameter": float(particles.diameter[idx]),
            "density": float(particles.density[idx]),
            "mass": float(particles.mass[idx]),
            "charge": float(particles.charge[idx]),
            "dep_particle_rel_permittivity": float(particles.dep_particle_rel_permittivity[idx]),
            "thermophoretic_coeff": float(particles.thermophoretic_coeff[idx]),
        }
    return defaults


def _row_particle_value(row, defaults: dict[str, float], *names: str, default: float = float("nan")) -> float:
    for name in names:
        if name in row:
            value = finite_float(row[name], default=float("nan"))
            if np.isfinite(value):
                return float(value)
    for name in names:
        if name in defaults:
            return float(defaults[name])
    return float(default)


def _sample_acceleration(prepared, points) -> pd.DataFrame:
    runtime = prepared.runtime
    force_runtime = force_runtime_parameters_from_catalog(runtime.force_catalog)
    compiled = compile_runtime_backend(
        runtime,
        int(runtime.spatial_dim),
        runtime.particles,
        force_runtime=force_runtime,
    )
    particle_defaults = _particle_defaults(runtime)
    fallback_defaults = next(iter(particle_defaults.values()), {})
    rows = []
    for fallback_idx, row in points.iterrows():
        point_id = row_point_id(row, int(fallback_idx))
        particle_id = row_particle_id(row)
        defaults = particle_defaults.get(int(particle_id), fallback_defaults) if particle_id is not None else fallback_defaults
        pos = row_position(row, runtime.spatial_dim)
        vel = row_velocity(row, runtime.spatial_dim)
        t_eval = row_time(row)
        mass = _row_particle_value(row, defaults, "mass", "particle_mass", default=float("nan"))
        charge = _row_particle_value(row, defaults, "charge", "particle_charge", default=0.0)
        q_over_m = charge / mass if np.isfinite(mass) and abs(mass) > 1.0e-300 else None
        acc = sample_compiled_acceleration_vector(
            compiled,
            int(runtime.spatial_dim),
            float(t_eval),
            pos,
            electric_q_over_m=q_over_m,
            force_runtime=force_runtime,
            particle_diameter=_row_particle_value(row, defaults, "diameter", "particle_diameter", default=0.0),
            particle_density=_row_particle_value(row, defaults, "density", "particle_density", default=0.0),
            particle_mass=mass if np.isfinite(mass) else None,
            dep_particle_rel_permittivity=_row_particle_value(
                row,
                defaults,
                "dep_particle_rel_permittivity",
                "particle_rel_permittivity",
                default=float("nan"),
            ),
            thermophoretic_coeff=_row_particle_value(row, defaults, "thermophoretic_coeff", default=float("nan")),
            velocity=vel,
            gas_density_kgm3=float(runtime.gas.density_kgm3),
            gas_mu_pas=float(runtime.gas.dynamic_viscosity_Pas),
            gas_temperature_K=float(runtime.gas.temperature),
            gas_molecular_mass_kg=float(runtime.gas.molecular_mass_amu) * 1.66053906660e-27,
        )
        for idx, component in enumerate(component_labels(runtime.spatial_dim)):
            rows.append(
                {
                    "point_id": point_id,
                    "particle_id": particle_id if particle_id is not None else "",
                    "time": float(t_eval),
                    "component": component,
                    "python_value": float(acc[idx]),
                    "physical_quantity": "acceleration",
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Python acceleration samples against COMSOL CSV samples.")
    parser.add_argument("--config", required=True, type=Path, help="particle_tracer_unified run YAML")
    parser.add_argument("--points", required=True, type=Path, help="CSV with sample points and optional particle properties")
    parser.add_argument("--comsol", "--reference", dest="reference", type=Path, default=None)
    parser.add_argument("--output", "--out", dest="output", type=Path, default=Path("acceleration_error.csv"))
    args = parser.parse_args()

    prepared = build_prepared_runtime_from_yaml(args.config)
    sampled = _sample_acceleration(prepared, read_csv(args.points))
    output = merge_with_reference(sampled, read_csv(args.reference)) if args.reference else sampled
    write_csv(output, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
