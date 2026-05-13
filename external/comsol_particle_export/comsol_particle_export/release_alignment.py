from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _first_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    lower = {str(c).strip().lower(): str(c) for c in frame.columns}
    for alias in aliases:
        found = lower.get(str(alias).strip().lower())
        if found is not None:
            return found
    return None


def _numeric(frame: pd.DataFrame, aliases: Iterable[str]) -> np.ndarray:
    col = _first_column(frame, aliases)
    if col is None:
        return np.full(len(frame), np.nan, dtype=np.float64)
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)


def _finite_summary(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p99": float(np.percentile(finite, 99.0)),
    }


def _particle_id(frame: pd.DataFrame, table_name: str) -> np.ndarray:
    col = _first_column(frame, ("particle_id", "ParticleID", "id", "pid", "particle"))
    if col is None:
        raise ValueError(f"{table_name} is missing a particle_id column")
    return pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=np.int64)


def _vector_frame(frame: pd.DataFrame, aliases: dict[str, tuple[str, ...]]) -> pd.DataFrame:
    out: dict[str, np.ndarray] = {}
    for name, names in aliases.items():
        values = _numeric(frame, names)
        if np.isfinite(values).any():
            out[name] = values
    return pd.DataFrame(out)


def _norm_error(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    common = [c for c in left.columns if c in right.columns]
    out = np.full(min(len(left), len(right)), np.nan, dtype=np.float64)
    if not common:
        return out
    a = left[common].to_numpy(dtype=np.float64)
    b = right[common].to_numpy(dtype=np.float64)
    valid = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
    out[valid] = np.linalg.norm(a[valid] - b[valid], axis=1)
    return out


def _property_error(matched: pd.DataFrame, name: str, aliases: tuple[str, ...]) -> np.ndarray:
    solver = _numeric(
        matched[[c for c in matched.columns if c.endswith("_solver")]].rename(columns=lambda c: c.removesuffix("_solver")),
        aliases,
    )
    comsol = _numeric(
        matched[[c for c in matched.columns if c.endswith("_comsol")]].rename(columns=lambda c: c.removesuffix("_comsol")),
        aliases,
    )
    del name
    return np.abs(solver - comsol)


def compare_release_tables(
    solver_particles_csv: str | Path,
    comsol_release_csv: str | Path,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    solver_path = Path(solver_particles_csv)
    comsol_path = Path(comsol_release_csv)
    solver = pd.read_csv(solver_path).copy()
    comsol = pd.read_csv(comsol_path).copy()
    solver["_particle_id"] = _particle_id(solver, str(solver_path))
    comsol["_particle_id"] = _particle_id(comsol, str(comsol_path))
    solver_prefixed = solver.rename(columns={col: f"{col}_solver" for col in solver.columns if col != "_particle_id"})
    comsol_prefixed = comsol.rename(columns={col: f"{col}_comsol" for col in comsol.columns if col != "_particle_id"})
    matched = solver_prefixed.merge(comsol_prefixed, on="_particle_id", how="inner")
    if matched.empty:
        raise ValueError("No matching particle_id values between solver particles and COMSOL release table")

    solver_cols = matched[[c for c in matched.columns if c.endswith("_solver")]].rename(columns=lambda c: c.removesuffix("_solver"))
    comsol_cols = matched[[c for c in matched.columns if c.endswith("_comsol")]].rename(columns=lambda c: c.removesuffix("_comsol"))
    solver_time = _numeric(solver_cols, ("release_time", "release_time_s", "t_release", "t0", "time_s", "time"))
    comsol_time = _numeric(comsol_cols, ("release_time", "release_time_s", "t_release", "t0", "time_s", "time"))
    position_aliases = {
        "x": ("x", "x_m", "x0", "x0_m", "r", "r_m", "r0", "r0_m"),
        "y": ("y", "y_m", "y0", "y0_m", "z", "z_m", "z0", "z0_m"),
        "z": ("z3", "z3_m", "z_3d", "z_3d_m"),
    }
    velocity_aliases = {
        "v_x": ("v_x", "vx", "v_x0", "vx0", "vr", "vr0"),
        "v_y": ("v_y", "vy", "v_y0", "vy0", "vz", "vz0"),
        "v_z": ("v_z", "vz3", "v_z0", "vz30"),
    }
    solver_pos = _vector_frame(solver_cols, position_aliases)
    comsol_pos = _vector_frame(comsol_cols, position_aliases)
    solver_vel = _vector_frame(solver_cols, velocity_aliases)
    comsol_vel = _vector_frame(comsol_cols, velocity_aliases)
    solver_source = _numeric(solver_cols, ("source_part_id", "source_entity", "source_boundary_id", "boundary_id", "part_id"))
    comsol_source = _numeric(comsol_cols, ("source_part_id", "source_entity", "source_boundary_id", "boundary_id", "part_id"))
    source_available = bool(np.isfinite(solver_source).any() and np.isfinite(comsol_source).any())
    source_valid = np.isfinite(solver_source) & np.isfinite(comsol_source)

    errors = pd.DataFrame(
        {
            "particle_id": matched["_particle_id"].to_numpy(dtype=np.int64),
            "release_time_error_s": np.abs(solver_time - comsol_time),
            "release_position_error_m": _norm_error(solver_pos, comsol_pos),
            "release_velocity_error_mps": _norm_error(solver_vel, comsol_vel),
            "diameter_error_m": _property_error(matched, "diameter", ("diameter", "diameter_m", "dp", "d")),
            "density_error_kgm3": _property_error(matched, "density", ("density", "density_kgm3", "rho_p", "rhop")),
            "mass_error_kg": _property_error(matched, "mass", ("mass", "mass_kg", "mp")),
            "charge_error_C": _property_error(matched, "charge", ("charge", "charge_C", "q")),
        }
    )
    if source_available:
        source_match = np.full(len(matched), False, dtype=bool)
        source_match[source_valid] = solver_source[source_valid].astype(int) == comsol_source[source_valid].astype(int)
        errors["source_entity_match"] = source_match
    summary = {
        "solver_particles_csv": str(solver_path),
        "comsol_release_csv": str(comsol_path),
        "solver_particle_count": int(len(solver)),
        "comsol_release_count": int(len(comsol)),
        "matched_particle_count": int(len(matched)),
        "unmatched_solver_count": int(len(solver) - len(matched)),
        "unmatched_comsol_count": int(len(comsol) - len(matched)),
        "release_time_error_s": _finite_summary(errors["release_time_error_s"].to_numpy(dtype=np.float64)),
        "release_position_error_m": _finite_summary(errors["release_position_error_m"].to_numpy(dtype=np.float64)),
        "release_velocity_error_mps": _finite_summary(errors["release_velocity_error_mps"].to_numpy(dtype=np.float64)),
        "diameter_error_m": _finite_summary(errors["diameter_error_m"].to_numpy(dtype=np.float64)),
        "density_error_kgm3": _finite_summary(errors["density_error_kgm3"].to_numpy(dtype=np.float64)),
        "mass_error_kg": _finite_summary(errors["mass_error_kg"].to_numpy(dtype=np.float64)),
        "charge_error_C": _finite_summary(errors["charge_error_C"].to_numpy(dtype=np.float64)),
        "source_entity_available": source_available,
        "source_entity_match_ratio": (
            float(np.mean(errors.loc[source_valid, "source_entity_match"].to_numpy(dtype=bool)))
            if source_available and "source_entity_match" in errors
            else None
        ),
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        errors.to_csv(out / "matched_release_errors.csv", index=False)
        (out / "release_alignment_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summary["outputs"] = {
            "matched_release_errors_csv": str(out / "matched_release_errors.csv"),
            "release_alignment_summary_json": str(out / "release_alignment_summary.json"),
        }
    return summary


__all__ = ("compare_release_tables",)
