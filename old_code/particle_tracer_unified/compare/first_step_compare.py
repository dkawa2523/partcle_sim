"""Orchestrate first-step comparisons and their command-line interface."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from particle_tracer_unified.application import load_case
from particle_tracer_unified.compare import _first_step_forces as _forces
from particle_tracer_unified.compare import _first_step_metrics as _metrics
from particle_tracer_unified.compare import _first_step_report as _report
from particle_tracer_unified.compare._common import json_safe, write_csv
from particle_tracer_unified.core.coordinate_systems import (
    axis_names_for_coordinate_system,
)
from particle_tracer_unified.core.datamodel import SolverContext
from particle_tracer_unified.solvers.high_fidelity_runtime import simulate_context


def _one_step_context(
    context: SolverContext,
    *,
    stochastic_policy: str,
    seed: int | None,
    dt: float | None = None,
) -> tuple[SolverContext, list[str]]:
    notes: list[str] = []
    dt_value = float(context.plan.dt if dt is None else dt)
    if not np.isfinite(dt_value) or dt_value <= 0.0:
        raise ValueError("solver.dt must be finite and > 0")
    stochastic_motion = context.options.stochastic_motion
    if seed is not None:
        stochastic_motion = replace(stochastic_motion, seed=int(seed))
    if str(stochastic_policy) == "off":
        stochastic_motion = replace(stochastic_motion, enabled=False)
        notes.append("stochastic motion disabled for deterministic first-step compare")
    output_plan = replace(context.plan.output, mode="debug", save_every=1)
    plan = replace(
        context.plan,
        dt=dt_value,
        t_end=dt_value,
        base_save_every=1,
        rng_seed=int(context.plan.rng_seed if seed is None else seed),
        output=output_plan,
    )
    options = replace(context.options, stochastic_motion=stochastic_motion)
    return replace(context, plan=plan, options=options), notes


def _parse_dt_sweep(value: str | None) -> list[float]:
    if value is None or not str(value).strip():
        return []
    values: list[float] = []
    for item in str(value).replace(";", ",").split(","):
        text = item.strip()
        if not text:
            continue
        dt = float(text)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt sweep values must be finite and > 0")
        values.append(dt)
    return values


def _run_one_step_compare(
    *,
    base_context: SolverContext,
    config_path: Path,
    output_dir: Path,
    reference: Path | None,
    stochastic: str,
    seed: int | None,
    dt: float | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    context, notes = _one_step_context(
        base_context,
        stochastic_policy=stochastic,
        seed=seed,
        dt=dt,
    )
    axes = axis_names_for_coordinate_system(
        context.coordinate_system, context.spatial_dim
    )
    dt_value = float(context.plan.dt)

    force_frame = _forces._force_contribution_frame(context)
    force_path = output_dir / "force_contributions.csv"
    write_csv(force_frame, force_path)

    payload = simulate_context(context, capture_debug=True)
    final_particles = pd.DataFrame({"particle_id": context.particles.particle_id})
    for axis_index, axis in enumerate(axes):
        final_particles[f"{axis}_m"] = np.asarray(payload.final_position)[:, axis_index]
        final_particles[f"v{axis}_mps"] = np.asarray(payload.final_velocity)[
            :, axis_index
        ]
    first_step_frame = _metrics._first_step_error_frame(
        context, final_particles, reference=reference
    )
    first_step_frame = _metrics._with_force_total_update_consistency(
        first_step_frame,
        force_frame,
        axes=axes,
        dt=dt_value,
    )
    first_step_path = output_dir / "first_step_error.csv"
    write_csv(first_step_frame, first_step_path)

    summary = _report._build_summary(
        config_path=config_path,
        output_dir=output_dir,
        context=context,
        force_frame=force_frame,
        first_step_frame=first_step_frame,
        force_path=force_path,
        first_step_path=first_step_path,
        reference=reference,
        stochastic=stochastic,
        seed=seed,
        notes=notes,
    )
    summary["solver_dt_s"] = dt_value
    summary["forced_t_end_s"] = dt_value
    _report._write_summary(summary, output_dir)
    return summary


def run_first_step_compare(
    *,
    config_path: Path,
    output_dir: Path,
    reference: Path | None = None,
    stochastic: str = "off",
    seed: int | None = None,
    dt_sweep: Sequence[float] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_context = load_case(config_path).solver_context
    summary = _run_one_step_compare(
        base_context=base_context,
        config_path=config_path,
        output_dir=output_dir,
        reference=reference,
        stochastic=stochastic,
        seed=seed,
    )
    sweep_values = [float(value) for value in (dt_sweep or [])]
    if sweep_values:
        runs: list[dict[str, Any]] = []
        sweep_root = output_dir / "dt_sweep"
        for index, dt in enumerate(sweep_values):
            dt_summary = _run_one_step_compare(
                base_context=base_context,
                config_path=config_path,
                output_dir=sweep_root / f"dt_{index:03d}",
                reference=reference,
                stochastic=stochastic,
                seed=seed,
                dt=dt,
            )
            runs.append(_report._dt_sweep_row(index, dt, dt_summary))
        _report._add_dt_sweep_ratios(runs)
        dt_sweep_summary = {
            "config": str(config_path),
            "stochastic_policy": str(stochastic),
            "seed": None if seed is None else int(seed),
            "dt_values_s": sweep_values,
            "runs": runs,
            "interpretation": (
                "For deterministic simple cases, force_update_* residuals should "
                "remain near floating-point roundoff when the local integrator "
                "assumptions apply. force_euler_* residuals should usually shrink "
                "as dt shrinks for relaxation cases. If neither improves, "
                "investigate force model, field sampling, initial velocity/release "
                "normal, or stochastic settings before tuning endpoint counts."
            ),
        }
        dt_sweep_path = output_dir / "dt_sweep_summary.json"
        dt_sweep_path.write_text(
            json.dumps(json_safe(dt_sweep_summary), indent=2) + "\n",
            encoding="utf-8",
        )
        summary["artifacts"]["dt_sweep_summary_json"] = str(dt_sweep_path)
        summary["dt_sweep"] = dt_sweep_summary
        _report._write_summary(summary, output_dir)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="particle-tracer compare first-step",
        description="Run a deterministic first-step and force-contribution comparison.",
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="particle_tracer_unified run YAML"
    )
    parser.add_argument(
        "--reference", type=Path, default=None, help="Optional first-step reference CSV"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("first_step_compare"))
    parser.add_argument("--stochastic", choices=("off", "from-config"), default="off")
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional deterministic seed override"
    )
    parser.add_argument(
        "--dt-sweep",
        default=None,
        help="Optional comma-separated dt values for one-step sensitivity runs",
    )
    args = parser.parse_args(argv)

    run_first_step_compare(
        config_path=args.config,
        output_dir=args.output_dir,
        reference=args.reference,
        stochastic=args.stochastic,
        seed=args.seed,
        dt_sweep=_parse_dt_sweep(args.dt_sweep),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
