"""Build first-step comparison tables and numerical error metrics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from particle_tracer_unified.compare._common import first_present
from particle_tracer_unified.core.coordinate_systems import (
    axis_names_for_coordinate_system,
)
from particle_tracer_unified.core.datamodel import ParticleTable, SolverContext
from particle_tracer_unified.solvers.integrator_common import (
    advance_state_2d,
    advance_state_3d,
)


class _ForceUpdateInputs(NamedTuple):
    initial_position: np.ndarray
    initial_velocity: np.ndarray
    total_acceleration: np.ndarray
    drag_acceleration: np.ndarray
    solver_position: np.ndarray
    solver_velocity: np.ndarray


class _ForceUpdatePrediction(NamedTuple):
    position: np.ndarray
    velocity: np.ndarray
    euler_position: np.ndarray
    euler_velocity: np.ndarray


def _solver_position_value(row: Mapping[str, Any], axis: str) -> float:
    candidate = f"{axis}_m"
    if candidate in row and pd.notna(row[candidate]):
        return float(row[candidate])
    return float("nan")


def _solver_velocity_value(row: Mapping[str, Any], axis: str) -> float:
    candidate = f"v{axis}_mps"
    if candidate in row and pd.notna(row[candidate]):
        return float(row[candidate])
    return float("nan")


def _reference_frame(
    reference: Path | None, axes: tuple[str, ...]
) -> pd.DataFrame | None:
    if reference is None:
        return None
    raw = pd.read_csv(reference)
    if "particle_id" not in raw.columns:
        raise ValueError("reference CSV must contain particle_id")
    rows = []
    for _, row in raw.iterrows():
        item: dict[str, Any] = {"particle_id": int(row["particle_id"])}
        for axis in axes:
            position_names = (f"{axis}1_ref", f"{axis}_ref", f"{axis}1", axis)
            velocity_names = (
                f"v{axis}1_ref",
                f"v_{axis}1_ref",
                f"v{axis}_ref",
                f"v_{axis}_ref",
                f"v{axis}",
                f"v_{axis}",
            )
            item[f"{axis}1_ref"] = float(first_present(row, position_names))
            item[f"v{axis}1_ref"] = float(first_present(row, velocity_names))
        rows.append(item)
    return pd.DataFrame(rows)


def _final_particle_mapping(
    final_by_id: pd.DataFrame,
    particle_id: int,
) -> Mapping[str, Any]:
    if particle_id not in final_by_id.index:
        return {}
    final: Any = final_by_id.loc[particle_id]
    return final if isinstance(final, Mapping) else final.to_dict()


def _first_step_particle_row(
    particles: ParticleTable,
    final_by_id: pd.DataFrame,
    particle_index: int,
    axes: tuple[str, ...],
) -> dict[str, Any]:
    particle_id = int(particles.particle_id[particle_index])
    final = _final_particle_mapping(final_by_id, particle_id)
    row: dict[str, Any] = {
        "particle_id": particle_id,
        "source_part_id": int(particles.source_part_id[particle_index]),
        "field_status": "",
        "notes": "",
    }
    for axis_index, axis in enumerate(axes):
        row[f"{axis}0"] = float(particles.position[particle_index, axis_index])
        row[f"v{axis}0"] = float(particles.velocity[particle_index, axis_index])
        row[f"{axis}1_solver"] = _solver_position_value(final, axis)
        row[f"v{axis}1_solver"] = _solver_velocity_value(final, axis)
        row[f"{axis}1_ref"] = float("nan")
        row[f"v{axis}1_ref"] = float("nan")
    return row


def _merge_first_step_reference(
    frame: pd.DataFrame,
    reference: Path | None,
    axes: tuple[str, ...],
) -> pd.DataFrame:
    reference_frame = _reference_frame(reference, axes)
    if reference_frame is None:
        return frame
    reference_columns = [
        column
        for column in reference_frame.columns
        if column in frame.columns and column != "particle_id"
    ]
    return frame.drop(columns=reference_columns).merge(
        reference_frame,
        on="particle_id",
        how="left",
    )


def _finite_vector_rows(
    solver_values: np.ndarray,
    reference_values: np.ndarray,
) -> np.ndarray:
    return np.all(np.isfinite(solver_values), axis=1) & np.all(
        np.isfinite(reference_values), axis=1
    )


def _vector_norm_errors(
    solver_values: np.ndarray,
    reference_values: np.ndarray,
    valid_rows: np.ndarray,
) -> np.ndarray:
    errors = np.full(solver_values.shape[0], np.nan, dtype=np.float64)
    errors[valid_rows] = np.linalg.norm(
        solver_values[valid_rows] - reference_values[valid_rows],
        axis=1,
    )
    return errors


def _velocity_speed_ratios(
    solver_velocity: np.ndarray,
    reference_velocity: np.ndarray,
    valid_rows: np.ndarray,
) -> np.ndarray:
    solver_speed = np.linalg.norm(solver_velocity, axis=1)
    reference_speed = np.linalg.norm(reference_velocity, axis=1)
    speed_valid = valid_rows & (reference_speed > 1.0e-300)
    ratios = np.full(solver_velocity.shape[0], np.nan, dtype=np.float64)
    ratios[speed_valid] = solver_speed[speed_valid] / reference_speed[speed_valid]
    return ratios


def _first_step_error_metrics(
    frame: pd.DataFrame,
    axes: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    solver_position = frame[[f"{axis}1_solver" for axis in axes]].to_numpy(
        dtype=np.float64
    )
    reference_position = frame[[f"{axis}1_ref" for axis in axes]].to_numpy(
        dtype=np.float64
    )
    solver_velocity = frame[[f"v{axis}1_solver" for axis in axes]].to_numpy(
        dtype=np.float64
    )
    reference_velocity = frame[[f"v{axis}1_ref" for axis in axes]].to_numpy(
        dtype=np.float64
    )
    valid_position = _finite_vector_rows(solver_position, reference_position)
    valid_velocity = _finite_vector_rows(solver_velocity, reference_velocity)
    return (
        _vector_norm_errors(solver_position, reference_position, valid_position),
        _vector_norm_errors(solver_velocity, reference_velocity, valid_velocity),
        _velocity_speed_ratios(
            solver_velocity,
            reference_velocity,
            valid_velocity,
        ),
    )


def _first_step_error_frame(
    context: SolverContext,
    final_particles: pd.DataFrame,
    *,
    reference: Path | None,
) -> pd.DataFrame:
    particles = context.particles
    axes = axis_names_for_coordinate_system(
        context.coordinate_system, context.spatial_dim
    )
    final_by_id = final_particles.set_index("particle_id", drop=False)
    rows = [
        _first_step_particle_row(particles, final_by_id, index, axes)
        for index in range(particles.count)
    ]
    out = _merge_first_step_reference(pd.DataFrame(rows), reference, axes)
    position_error, velocity_error, speed_ratio = _first_step_error_metrics(out, axes)
    out["position_error_m"] = position_error
    out["velocity_error_mps"] = velocity_error
    out["speed_ratio"] = speed_ratio
    return out


def _force_update_inputs(
    row: Any,
    force_row: Mapping[str, Any],
    axes: tuple[str, ...],
) -> _ForceUpdateInputs:
    initial_position: list[float] = []
    initial_velocity: list[float] = []
    total_acceleration: list[float] = []
    drag_acceleration: list[float] = []
    solver_position: list[float] = []
    solver_velocity: list[float] = []
    for axis in axes:
        initial_position.append(float(row.get(f"{axis}0", np.nan)))
        initial_velocity.append(float(row.get(f"v{axis}0", np.nan)))
        total_acceleration.append(float(force_row.get(f"total_a{axis}", np.nan)))
        drag_acceleration.append(float(force_row.get(f"drag_a{axis}", np.nan)))
        solver_position.append(float(row.get(f"{axis}1_solver", np.nan)))
        solver_velocity.append(float(row.get(f"v{axis}1_solver", np.nan)))
    return _ForceUpdateInputs(
        np.asarray(initial_position, dtype=np.float64),
        np.asarray(initial_velocity, dtype=np.float64),
        np.asarray(total_acceleration, dtype=np.float64),
        np.asarray(drag_acceleration, dtype=np.float64),
        np.asarray(solver_position, dtype=np.float64),
        np.asarray(solver_velocity, dtype=np.float64),
    )


def _has_drag_predictor_inputs(
    inputs: _ForceUpdateInputs,
    axis_count: int,
    tau_eff: float,
) -> bool:
    return bool(
        axis_count in (2, 3)
        and np.all(np.isfinite(inputs.initial_position))
        and np.all(np.isfinite(inputs.initial_velocity))
        and np.all(np.isfinite(inputs.total_acceleration))
        and np.all(np.isfinite(inputs.drag_acceleration))
        and np.isfinite(tau_eff)
        and tau_eff > 0.0
    )


def _drag_predictor_2d(
    inputs: _ForceUpdateInputs,
    target: np.ndarray,
    body_acceleration: np.ndarray,
    tau_eff: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, vx1, vy1 = advance_state_2d(
        float(inputs.initial_position[0]),
        float(inputs.initial_position[1]),
        float(inputs.initial_velocity[0]),
        float(inputs.initial_velocity[1]),
        float(target[0]),
        float(target[1]),
        float(body_acceleration[0]),
        float(body_acceleration[1]),
        float(tau_eff),
        float(dt),
    )
    return (
        np.asarray([x1, y1], dtype=np.float64),
        np.asarray([vx1, vy1], dtype=np.float64),
    )


def _drag_predictor_3d(
    inputs: _ForceUpdateInputs,
    target: np.ndarray,
    body_acceleration: np.ndarray,
    tau_eff: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, z1, vx1, vy1, vz1 = advance_state_3d(
        float(inputs.initial_position[0]),
        float(inputs.initial_position[1]),
        float(inputs.initial_position[2]),
        float(inputs.initial_velocity[0]),
        float(inputs.initial_velocity[1]),
        float(inputs.initial_velocity[2]),
        float(target[0]),
        float(target[1]),
        float(target[2]),
        float(body_acceleration[0]),
        float(body_acceleration[1]),
        float(body_acceleration[2]),
        float(tau_eff),
        float(dt),
    )
    return (
        np.asarray([x1, y1, z1], dtype=np.float64),
        np.asarray([vx1, vy1, vz1], dtype=np.float64),
    )


def _force_update_prediction(
    inputs: _ForceUpdateInputs,
    force_row: Mapping[str, Any],
    axes: tuple[str, ...],
    dt: float,
) -> _ForceUpdatePrediction:
    euler_velocity = inputs.initial_velocity + inputs.total_acceleration * float(dt)
    euler_position = inputs.initial_position + euler_velocity * float(dt)
    tau_eff = float(force_row.get("drag_tau_eff_s", np.nan))
    position = euler_position.copy()
    velocity = euler_velocity.copy()
    if _has_drag_predictor_inputs(inputs, len(axes), tau_eff):
        target = inputs.initial_velocity + inputs.drag_acceleration * float(tau_eff)
        body_acceleration = inputs.total_acceleration - inputs.drag_acceleration
        predictor = _drag_predictor_2d if len(axes) == 2 else _drag_predictor_3d
        position, velocity = predictor(
            inputs,
            target,
            body_acceleration,
            tau_eff,
            dt,
        )
    return _ForceUpdatePrediction(
        position,
        velocity,
        euler_position,
        euler_velocity,
    )


def _write_force_update_prediction(
    frame: pd.DataFrame,
    row_index: Any,
    axes: tuple[str, ...],
    prediction: _ForceUpdatePrediction,
) -> None:
    for axis_index, axis in enumerate(axes):
        frame.loc[row_index, f"{axis}1_force_total"] = float(
            prediction.position[axis_index]
        )
        frame.loc[row_index, f"v{axis}1_force_total"] = float(
            prediction.velocity[axis_index]
        )
        frame.loc[row_index, f"{axis}1_force_total_euler"] = float(
            prediction.euler_position[axis_index]
        )
        frame.loc[row_index, f"v{axis}1_force_total_euler"] = float(
            prediction.euler_velocity[axis_index]
        )


def _finite_residual(prediction: np.ndarray, solver_value: np.ndarray) -> float:
    if np.all(np.isfinite(prediction)) and np.all(np.isfinite(solver_value)):
        return float(np.linalg.norm(solver_value - prediction))
    return float("nan")


def _with_force_total_update_consistency(
    first_step_frame: pd.DataFrame,
    force_frame: pd.DataFrame,
    *,
    axes: tuple[str, ...],
    dt: float,
) -> pd.DataFrame:
    out = first_step_frame.copy()
    if "particle_id" not in force_frame.columns:
        raise ValueError("force contribution frame must contain particle_id")
    force_by_id = force_frame.set_index("particle_id", drop=False)
    velocity_residual = np.full(len(out), np.nan, dtype=np.float64)
    position_residual = np.full(len(out), np.nan, dtype=np.float64)
    euler_velocity_residual = np.full(len(out), np.nan, dtype=np.float64)
    euler_position_residual = np.full(len(out), np.nan, dtype=np.float64)
    for row_index, row in out.iterrows():
        pid = int(row["particle_id"])
        if pid not in force_by_id.index:
            continue
        force_row: Any = force_by_id.loc[pid]
        residual_index: Any = row_index
        inputs = _force_update_inputs(row, force_row, axes)
        prediction = _force_update_prediction(inputs, force_row, axes, dt)
        _write_force_update_prediction(out, row_index, axes, prediction)
        velocity_residual[residual_index] = _finite_residual(
            prediction.velocity,
            inputs.solver_velocity,
        )
        position_residual[residual_index] = _finite_residual(
            prediction.position,
            inputs.solver_position,
        )
        euler_velocity_residual[residual_index] = _finite_residual(
            prediction.euler_velocity,
            inputs.solver_velocity,
        )
        euler_position_residual[residual_index] = _finite_residual(
            prediction.euler_position,
            inputs.solver_position,
        )
    out["force_total_update_velocity_residual_mps"] = velocity_residual
    out["force_total_update_position_residual_m"] = position_residual
    out["force_total_euler_velocity_residual_mps"] = euler_velocity_residual
    out["force_total_euler_position_residual_m"] = euler_position_residual
    return out
