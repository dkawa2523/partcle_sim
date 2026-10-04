"""Build and serialize first-step and time-step sweep summaries."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from particle_tracer_unified.compare._common import finite_summary, json_safe
from particle_tracer_unified.core.coordinate_systems import (
    axis_names_for_coordinate_system,
)
from particle_tracer_unified.core.datamodel import SolverContext
from particle_tracer_unified.solvers.forces import force_catalog_summary

_DT_SWEEP_RATIO_COLUMNS = (
    (
        "force_update_velocity_residual_mps",
        "force_update_velocity_residual_max_ratio_vs_previous",
    ),
    (
        "force_update_position_residual_m",
        "force_update_position_residual_max_ratio_vs_previous",
    ),
    (
        "force_euler_velocity_residual_mps",
        "force_euler_velocity_residual_max_ratio_vs_previous",
    ),
    (
        "force_euler_position_residual_m",
        "force_euler_position_residual_max_ratio_vs_previous",
    ),
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _build_summary(
    *,
    config_path: Path,
    output_dir: Path,
    context: SolverContext,
    force_frame: pd.DataFrame,
    first_step_frame: pd.DataFrame,
    force_path: Path,
    first_step_path: Path,
    reference: Path | None,
    stochastic: str,
    seed: int | None,
    notes: list[str],
) -> dict[str, Any]:
    dim = int(context.spatial_dim)
    force_summary = force_catalog_summary(context.force_catalog)
    return {
        "config": str(config_path),
        "spatial_dim": int(dim),
        "coordinate_system": str(context.coordinate_system),
        "axis_names": list(
            axis_names_for_coordinate_system(
                context.coordinate_system, context.spatial_dim
            )
        ),
        "particle_count": len(first_step_frame),
        "reference_particle_count": int(
            0 if reference is None else len(pd.read_csv(reference))
        ),
        "compared_particle_count": int(
            np.count_nonzero(
                np.isfinite(first_step_frame["position_error_m"].to_numpy(dtype=float))
            )
        ),
        "stochastic_policy": str(stochastic),
        "stochastic_disabled_for_compare": int(str(stochastic) == "off"),
        "stochastic_controlled_by_seed": int(seed is not None),
        "seed": None if seed is None else int(seed),
        "solver_dt_s": float(context.plan.dt),
        "forced_t_end_s": float(context.plan.t_end),
        "enabled_forces": list(force_summary.get("enabled_forces", []))
        if isinstance(force_summary, Mapping)
        else [],
        "force_contribution_rows": len(force_frame),
        "position_error_m": finite_summary(
            first_step_frame["position_error_m"].to_numpy(dtype=float)
        ),
        "velocity_error_mps": finite_summary(
            first_step_frame["velocity_error_mps"].to_numpy(dtype=float)
        ),
        "force_total_update": {
            "velocity_residual_mps": finite_summary(
                first_step_frame["force_total_update_velocity_residual_mps"].to_numpy(
                    dtype=float
                )
            ),
            "position_residual_m": finite_summary(
                first_step_frame["force_total_update_position_residual_m"].to_numpy(
                    dtype=float
                )
            ),
            "interpretation": (
                "Near-zero residuals indicate that force_contributions total is "
                "compatible with the configured local one-step integrator under "
                "deterministic start-state field sampling assumptions. Non-zero "
                "residuals can indicate stochastic motion, changing fields, "
                "wall/contact behavior, or a force/field mismatch."
            ),
        },
        "force_total_euler": {
            "velocity_residual_mps": finite_summary(
                first_step_frame["force_total_euler_velocity_residual_mps"].to_numpy(
                    dtype=float
                )
            ),
            "position_residual_m": finite_summary(
                first_step_frame["force_total_euler_position_residual_m"].to_numpy(
                    dtype=float
                )
            ),
            "interpretation": (
                "Euler residuals compare the solver step with v1 = v0 + "
                "total_acceleration * dt and x1 = x0 + v1 * dt. Residuals that "
                "shrink with dt usually indicate expected integrator "
                "finite-step behavior rather than a force-total mismatch."
            ),
        },
        "artifacts": {
            "first_step_error_csv": str(first_step_path),
            "force_contributions_csv": str(force_path),
            "output_dir": str(output_dir),
        },
        "notes": notes,
    }


def _write_summary(summary: dict[str, Any], output_dir: Path) -> None:
    summary_path = output_dir / "first_step_summary.json"
    summary["artifacts"]["summary_json"] = str(summary_path)
    payload = json.dumps(json_safe(summary), indent=2) + "\n"
    summary_path.write_text(payload, encoding="utf-8")


def _dt_sweep_row(index: int, dt: float, summary: Mapping[str, Any]) -> dict[str, Any]:
    force_update = _mapping(summary.get("force_total_update", {}))
    force_euler = _mapping(summary.get("force_total_euler", {}))
    vel = _mapping(force_update.get("velocity_residual_mps", {}))
    pos = _mapping(force_update.get("position_residual_m", {}))
    euler_vel = _mapping(force_euler.get("velocity_residual_mps", {}))
    euler_pos = _mapping(force_euler.get("position_residual_m", {}))
    pos_ref = _mapping(summary.get("position_error_m", {}))
    vel_ref = _mapping(summary.get("velocity_error_mps", {}))
    return {
        "index": int(index),
        "dt_s": float(dt),
        "output_dir": str(summary.get("artifacts", {}).get("output_dir", "")),
        "force_update_velocity_residual_mps": vel,
        "force_update_position_residual_m": pos,
        "force_euler_velocity_residual_mps": euler_vel,
        "force_euler_position_residual_m": euler_pos,
        "reference_position_error_m": pos_ref,
        "reference_velocity_error_mps": vel_ref,
    }


def _dt_sweep_maxima(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        _mapping(row.get(summary_column, {})).get("max")
        for summary_column, _ in _DT_SWEEP_RATIO_COLUMNS
    )


def _ratio_vs_previous(current: Any, previous: float | None) -> float | None:
    return (
        float(current) / float(previous)
        if previous is not None and current is not None and float(previous) > 0.0
        else None
    )


def _add_dt_sweep_ratios(rows: list[dict[str, Any]]) -> None:
    previous_maxima: list[float | None] = [None] * len(_DT_SWEEP_RATIO_COLUMNS)
    for row in rows:
        maxima = _dt_sweep_maxima(row)
        for index, (_, ratio_column) in enumerate(_DT_SWEEP_RATIO_COLUMNS):
            row[ratio_column] = _ratio_vs_previous(
                maxima[index],
                previous_maxima[index],
            )
        for index, current in enumerate(maxima):
            if current is not None:
                previous_maxima[index] = float(current)
