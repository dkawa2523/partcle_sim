"""Validate solver time support and translate outcomes to public results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ._application_types import (
    RunStats,
    SimulationCase,
    SimulationResult,
    SimulationState,
)
from .core.coordinate_systems import axis_names_for_coordinate_system
from .experimental_features import enabled_experimental_features
from .force_models import force_parameter_mapping
from .solvers.diagnostics import invalid_stop_reason_names

if TYPE_CHECKING:
    from .solvers.forces import ForceCatalog


def _transient_time_axes(case: SimulationCase) -> list[tuple[str, np.ndarray]]:
    field_provider = case._context.field_provider
    if field_provider is None:
        raise ValueError("simulation case requires a field provider")
    field = field_provider.field
    if str(getattr(field, "time_mode", "steady")) != "transient":
        return []
    axes = []
    for name, series in getattr(field, "quantities", {}).items():
        times = np.asarray(getattr(series, "times", ()), dtype=np.float64)
        if times.size > 1:
            axes.append((str(name), times))
    if not axes:
        raise ValueError(
            "transient field has no quantity with a multi-sample time axis"
        )
    return axes


def require_transient_field_time_support(case: SimulationCase) -> None:
    """Ensure transient fields cover every particle integration interval."""

    axes = _transient_time_axes(case)
    if not axes:
        return
    support_start = max(float(times[0]) for _name, times in axes)
    support_end = min(float(times[-1]) for _name, times in axes)
    if support_end < support_start:
        names = [name for name, _times in axes]
        raise ValueError(
            "transient field quantities have no common time support: "
            f"quantities={names}"
        )
    t_end = float(case.config.time.t_end)
    release_times = np.asarray(case._context.particles.release_time, dtype=np.float64)
    integrated = release_times[release_times < t_end]
    if integrated.size == 0:
        return
    required_start = float(np.min(integrated))
    if support_start > required_start or support_end < t_end:
        raise ValueError(
            "transient field time support does not cover the required particle "
            "integration interval: "
            f"field_support_s=[{support_start}, {support_end}], "
            f"required_support_s=[{required_start}, {t_end}]"
        )


def _readonly_copy(value: Any, dtype: Any | None = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _terminal_state(payload: Any) -> np.ndarray:
    count = int(np.asarray(payload.final_position).shape[0])
    labels = np.full(count, "inactive", dtype=object)
    labels[np.asarray(payload.active, dtype=bool)] = "active_free_flight"
    labels[np.asarray(payload.contact_sliding, dtype=bool)] = "contact_sliding"
    labels[np.asarray(payload.contact_endpoint_stopped, dtype=bool)] = (
        "contact_endpoint_stopped"
    )
    for name, values in (
        ("invalid_mask_stopped", payload.invalid_mask_stopped),
        ("numerical_boundary_stopped", payload.numerical_boundary_stopped),
        ("stuck", payload.stuck),
        ("frozen", payload.frozen),
        ("absorbed", payload.absorbed),
        ("escaped", payload.escaped),
    ):
        labels[np.asarray(values, dtype=bool)] = name
    return labels


def _terminal_overlap_count(payload: Any) -> int:
    masks = np.vstack(
        [
            np.asarray(payload.active, dtype=bool),
            np.asarray(payload.invalid_mask_stopped, dtype=bool),
            np.asarray(payload.numerical_boundary_stopped, dtype=bool),
            np.asarray(payload.stuck, dtype=bool),
            np.asarray(payload.frozen, dtype=bool),
            np.asarray(payload.absorbed, dtype=bool),
            np.asarray(payload.escaped, dtype=bool),
        ]
    )
    return int(np.count_nonzero(np.sum(masks, axis=0) > 1))


def _nonfinite_row_count(values: np.ndarray) -> int:
    array = np.asarray(values, dtype=np.float64)
    return int(np.count_nonzero(~np.all(np.isfinite(array), axis=1)))


def force_contribution_rows(
    force_catalog: ForceCatalog,
) -> tuple[Mapping[str, Any], ...]:
    rows: list[Mapping[str, Any]] = []
    acceleration_forces = {
        "gravity",
        "electric",
        "thermophoresis",
        "dielectrophoresis",
        "lift",
        "pressure_gradient",
        "virtual_mass",
    }
    for binding in force_catalog.bindings:
        force = binding.force
        rows.append(
            {
                "name": force.name,
                "enabled": int(bool(force.enabled)),
                "model": force.model,
                "status": force.status,
                "physical_quantity": (
                    "acceleration" if force.name in acceleration_forces else "force"
                ),
                "required_fields": binding.required_fields,
                "optional_fields": binding.optional_fields,
                "field_sources": dict(binding.field_sources),
                "parameters": force_parameter_mapping(force),
            }
        )
    return tuple(rows)


def _wall_summary(payload: Any) -> dict[tuple[int, str, str], int]:
    return {
        (int(part_id), str(outcome), str(wall_law)): int(count)
        for (part_id, outcome, wall_law), count in payload.wall_summary_counts.items()
    }


def _simulation_state(
    case: SimulationCase,
    payload: Any,
    terminal_state: np.ndarray,
) -> SimulationState:
    particles = case._context.particles
    invalid_reason_code = np.asarray(payload.invalid_stop_reason_code, dtype=np.uint8)
    invalid_reason = np.asarray(
        invalid_stop_reason_names(invalid_reason_code), dtype=object
    )
    return SimulationState(
        particle_id=_readonly_copy(particles.particle_id, np.int64),
        position_m=_readonly_copy(payload.final_position, np.float64),
        velocity_mps=_readonly_copy(payload.final_velocity, np.float64),
        charge_C=_readonly_copy(payload.final_charge, np.float64),
        release_time_s=_readonly_copy(particles.release_time, np.float64),
        source_part_id=_readonly_copy(particles.source_part_id, np.int64),
        material_id=_readonly_copy(particles.material_id, np.int64),
        mass_kg=_readonly_copy(particles.mass, np.float64),
        drag_diameter_m=_readonly_copy(particles.diameter, np.float64),
        released=_readonly_copy(payload.released, bool),
        terminal_state=_readonly_copy(terminal_state, object),
        invalid_stop_reason_code=_readonly_copy(invalid_reason_code, np.uint8),
        invalid_stop_reason=_readonly_copy(invalid_reason, object),
        contact_part_id=_readonly_copy(payload.contact_part_id, np.int64),
        contact_normal=_readonly_copy(payload.contact_normal, np.float64),
    )


def _run_stats(
    case: SimulationCase,
    payload: Any,
    wall_summary: Mapping,
    terminal_state: np.ndarray,
) -> RunStats:
    terminal_counts = {
        str(name): int(np.count_nonzero(terminal_state == name))
        for name in sorted(set(map(str, terminal_state.tolist())))
    }
    wall_outcome_counts: dict[str, int] = {}
    for (_part_id, outcome, _wall_law), count in wall_summary.items():
        wall_outcome_counts[outcome] = wall_outcome_counts.get(outcome, 0) + int(count)
    diagnostics: Mapping[str, Any] = payload.collision_diagnostics
    safety_counters = {
        "nonfinite_position_count": _nonfinite_row_count(payload.final_position),
        "nonfinite_velocity_count": _nonfinite_row_count(payload.final_velocity),
        "field_support_exit_count": int(np.count_nonzero(payload.invalid_mask_stopped)),
        "unresolved_crossing_count": int(
            diagnostics.get("unresolved_crossing_count", 0)
        ),
        "max_hits_reached_count": int(diagnostics.get("max_hits_reached_count", 0)),
        "terminal_state_overlap_count": _terminal_overlap_count(payload),
        "wall_interaction_count": int(sum(wall_summary.values())),
    }
    particles = case._context.particles
    return RunStats(
        timing_s={str(key): float(value) for key, value in payload.timing_s.items()},
        memory_estimate_bytes={
            str(key): int(value) for key, value in payload.memory_estimate_bytes.items()
        },
        terminal_counts=terminal_counts,
        wall_outcome_counts=wall_outcome_counts,
        particle_count=int(particles.count),
        released_count=int(np.count_nonzero(payload.released)),
        safety_counters=safety_counters,
    )


def _debug_result(case: SimulationCase, payload: Any) -> dict[str, Any]:
    if case.config.output.mode != "debug":
        return {}
    if payload.debug is None:
        raise RuntimeError(
            "debug output was requested but the solver did not capture debug buffers"
        )
    solver_debug = payload.debug
    return {
        "trajectory_m": _readonly_copy(solver_debug.trajectory_positions, np.float64),
        "save_frames": tuple(dict(row) for row in solver_debug.save_frames),
        "wall_events": tuple(dict(row) for row in solver_debug.wall_events),
        "step_summary": tuple(dict(row) for row in solver_debug.step_summary),
        "force_contributions": force_contribution_rows(case._context.force_catalog),
        "collision_diagnostics": dict(payload.collision_diagnostics),
        "max_hit_events": tuple(dict(row) for row in solver_debug.max_hit_events),
    }


def build_simulation_result(case: SimulationCase, payload: Any) -> SimulationResult:
    """Translate the mutable solver outcome into the immutable public result."""

    wall_summary = _wall_summary(payload)
    terminal_state = _terminal_state(payload)
    return SimulationResult(
        plan=case.plan,
        state=_simulation_state(case, payload, terminal_state),
        stats=_run_stats(case, payload, wall_summary, terminal_state),
        wall_summary=wall_summary,
        axis_names=tuple(
            axis_names_for_coordinate_system(
                case.config.case.coordinate_system,
                case.config.case.spatial_dim,
            )
        ),
        drag_model=case.plan.drag_model,
        experimental_features=enabled_experimental_features(
            case._context.force_catalog.model,
            case.config.physics,
        ),
        final_step_name=str(payload.final_step_name),
        final_segment_name=str(payload.final_segment_name),
        execution_metadata=case._execution,
        debug=_debug_result(case, payload),
    )


__all__ = (
    "build_simulation_result",
    "force_contribution_rows",
    "require_transient_field_time_support",
)
