from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from field_backend_helpers import geometry_provider, regular_field_provider

from particle_tracer_unified import load_case, simulate, validate_case, write_result
from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
)
from particle_tracer_unified.solvers import high_fidelity_collision
from particle_tracer_unified.solvers.base_field_sampling import (
    sample_compiled_valid_mask_status,
)
from particle_tracer_unified.solvers.field_compilation import compile_runtime_backend
from particle_tracer_unified.solvers.integrator_common import DRAG_MODEL_NONE
from particle_tracer_unified.solvers.segment_motion import (
    SegmentMotionRequest,
    trace_motion_segment,
)
from particle_tracer_unified.solvers.segment_trace import TraceRefinementDecision

BOTTOM_PART_ID = 10
RIGHT_PART_ID = 20
TOP_PART_ID = 30
LEFT_PART_ID = 40


def _write_quarter_excursion_case(root: Path) -> Path:
    """Write a path that leaves the box only during the first half-step.

    The exact vertical trajectory is

        y(t) = 0.9 + 0.85 t - 1.7 t**2.

    It is outside the top wall at t=0.25, while both the ETD2 midpoint and
    endpoint are inside (y(0.5)=0.9 and y(1)=0.05).  Endpoint/midpoint-only
    collision tests therefore miss this wall crossing.
    """

    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "particle_id": 1,
                "x_m": 0.0,
                "y_m": 0.9,
                "vx_mps": 0.0,
                "vy_mps": 0.85,
                "release_time_s": 0.0,
                "mass_kg": 1.0e-12,
                "drag_diameter_m": 1.0e-6,
                "charge_C": 0.0,
                "source_part_id": BOTTOM_PART_ID,
            }
        ]
    ).to_csv(root / "particles.csv", index=False)

    laws = {
        BOTTOM_PART_ID: "specular",
        RIGHT_PART_ID: "specular",
        TOP_PART_ID: "absorb",
        LEFT_PART_ID: "specular",
    }
    pd.DataFrame(
        [
            {
                "part_id": part_id,
                "part_name": f"part_{part_id}",
                "role": "wall",
                "material_id": part_id,
                "material_name": "test_material",
                "wall_law": laws[part_id],
                "wall_stick_probability": 0.0,
                "wall_restitution": 1.0,
                "wall_diffuse_fraction": 0.0,
                "wall_critical_sticking_velocity_mps": 0.0,
            }
            for part_id in (BOTTOM_PART_ID, RIGHT_PART_ID, TOP_PART_ID, LEFT_PART_ID)
        ]
    ).to_csv(root / "boundaries.csv", index=False)

    config = {
        "schema_version": 2,
        "case": {
            "spatial_dim": 2,
            "coordinate_system": "cartesian_xy",
            "adapter": "native",
        },
        "inputs": {
            "particles": "particles.csv",
            "boundaries": "boundaries.csv",
            "geometry": {
                "kind": "box",
                "parameters": {
                    "bounds": [-1.0, 1.0, -1.0, 1.0],
                    "grid_shape": [21, 21],
                    # Synthetic-box edge order is bottom, right, top, left.
                    "boundary_part_ids": [
                        BOTTOM_PART_ID,
                        RIGHT_PART_ID,
                        TOP_PART_ID,
                        LEFT_PART_ID,
                    ],
                },
            },
            "field": {
                "kind": "linear_shear",
                "parameters": {
                    "shear_rate": 0.0,
                    "dynamic_viscosity_Pas": 1.8e-5,
                },
            },
        },
        "physics": {
            "drag": {"model": "none"},
            "gas": {},
            "forces": {
                "gravity": {
                    "enabled": True,
                    "model": "constant_acceleration",
                    "parameters": {
                        "acceleration_mps2": [0.0, -3.4],
                        "buoyancy": False,
                    },
                }
            },
            "seed": 31415,
        },
        "time": {"dt": 1.0, "t_end": 1.0},
        "output": {"mode": "debug", "trajectory_interval_steps": 1},
    }
    config_path = root / "case.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _scale_quarter_excursion_case(path: Path, scale: float) -> None:
    particles_path = path.parent / "particles.csv"
    particles = pd.read_csv(particles_path)
    for column in ("x_m", "y_m", "vx_mps", "vy_mps", "drag_diameter_m"):
        particles[column] = particles[column].astype(float) * float(scale)
    particles.to_csv(particles_path, index=False)

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    geometry = config["inputs"]["geometry"]["parameters"]
    geometry["bounds"] = [float(value) * float(scale) for value in geometry["bounds"]]
    acceleration = config["physics"]["forces"]["gravity"]["parameters"][
        "acceleration_mps2"
    ]
    config["physics"]["forces"]["gravity"]["parameters"]["acceleration_mps2"] = [
        float(value) * float(scale) for value in acceleration
    ]
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_over_refinement_curve_case(
    root: Path,
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Write a parabola whose narrow wall excursion first appears at 32 substeps."""

    path = _write_quarter_excursion_case(root)
    peak_time = 33.0 / 64.0
    curvature = 100.0
    peak_y = 1.01
    initial_position = np.asarray(
        [0.0, peak_y - curvature * peak_time * peak_time],
        dtype=np.float64,
    )
    initial_velocity = np.asarray([0.0, 2.0 * curvature * peak_time], dtype=np.float64)

    particles_path = path.parent / "particles.csv"
    particles = pd.read_csv(particles_path)
    particles.loc[0, ["x_m", "y_m"]] = initial_position
    particles.loc[0, ["vx_mps", "vy_mps"]] = initial_velocity
    particles.to_csv(particles_path, index=False)

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["inputs"]["geometry"]["parameters"].update(
        {"bounds": [-1.0, 1.0, -100.0, 1.0], "grid_shape": [21, 101]}
    )
    config["physics"]["forces"]["gravity"]["parameters"]["acceleration_mps2"] = [
        0.0,
        -2.0 * curvature,
    ]
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path, initial_position, initial_velocity


def _write_over_refinement_support_case(root: Path) -> Path:
    """Write a clean path whose support spacing requires more than 16 substeps."""

    path = _write_quarter_excursion_case(root)
    particles_path = path.parent / "particles.csv"
    particles = pd.read_csv(particles_path)
    particles.loc[0, ["x_m", "y_m", "vx_mps", "vy_mps"]] = [0.0, 0.0, 1.0, 0.0]
    particles.to_csv(particles_path, index=False)

    axis_x = np.linspace(-1.0, 2.0, 301, dtype=np.float64)
    axis_y = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
    shape = (axis_x.size, axis_y.size)
    valid_mask = np.ones(shape, dtype=bool)
    # Make support spacing relevant without placing the invalid cell on the
    # particle path.  A one-second, one-metre trace needs more than the fixed
    # 16-substep replay budget to rule out an unresolved narrow support island.
    valid_mask[0, 0] = False
    np.savez_compressed(
        path.parent / "field.npz",
        axis_0=axis_x,
        axis_1=axis_y,
        times=np.asarray([0.0], dtype=np.float64),
        valid_mask=valid_mask,
        ux=np.zeros(shape, dtype=np.float64),
        uy=np.zeros(shape, dtype=np.float64),
    )

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["inputs"]["geometry"]["parameters"].update(
        {"bounds": [-1.0, 2.0, -1.0, 1.0], "grid_shape": [301, 3]}
    )
    config["inputs"]["field"] = {"kind": "precomputed_npz", "path": "field.npz"}
    config["physics"]["forces"] = {}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_trace_refinement_limit_is_not_resolution() -> None:
    geometry = TraceRefinementDecision(
        geometry_risk=True,
        support_substeps=16,
        max_substeps=16,
    )
    support = TraceRefinementDecision(
        geometry_risk=False,
        support_substeps=17,
        max_substeps=16,
    )

    assert geometry.limit_reached(current_substeps=16)
    assert not geometry.resolved(current_substeps=16)
    assert support.limit_reached(current_substeps=16)
    assert not support.resolved(current_substeps=16)


def test_curve_requiring_more_than_sixteen_substeps_fails_closed(
    tmp_path: Path,
) -> None:
    path, initial_position, initial_velocity = _write_over_refinement_curve_case(
        tmp_path / "over-refinement-curve"
    )

    result = simulate(load_case(path))

    assert result.state.terminal_state.tolist() == ["numerical_boundary_stopped"]
    assert result.state.position_m[0].tolist() == pytest.approx(
        initial_position, abs=1.0e-14
    )
    assert result.state.velocity_mps[0].tolist() == pytest.approx(
        initial_velocity, abs=1.0e-14
    )
    assert result.stats.safety_counters["unresolved_crossing_count"] == 1
    assert result.stats.safety_counters["wall_interaction_count"] == 0
    diagnostics = result.debug["collision_diagnostics"]
    assert diagnostics["numerical_boundary_stop_count"] == 1
    assert diagnostics["numerical_boundary_stop_reason_counts"] == {
        "trace_refinement_unresolved": 1
    }


def test_support_requiring_more_than_sixteen_substeps_stops_invalid(
    tmp_path: Path,
) -> None:
    path = _write_over_refinement_support_case(tmp_path / "over-refinement-support")

    result = simulate(load_case(path))

    assert result.state.terminal_state.tolist() == ["invalid_mask_stopped"]
    assert result.state.position_m[0].tolist() == pytest.approx([0.0, 0.0], abs=1.0e-14)
    assert result.state.invalid_stop_reason.tolist() == [
        "freeflight_field_support_refinement_exhausted"
    ]
    assert result.stats.safety_counters["field_support_exit_count"] == 1
    assert result.stats.safety_counters["unresolved_crossing_count"] == 1


def test_quarter_substep_excursion_hits_top_terminal_wall(tmp_path: Path) -> None:
    path = _write_quarter_excursion_case(tmp_path / "quarter-excursion")
    case = load_case(path)

    report = validate_case(case)
    assert report.passed, report.errors

    result = simulate(case)

    assert result.state.terminal_state.tolist() == ["absorbed"]
    assert result.stats.terminal_counts == {"absorbed": 1}
    assert result.stats.wall_outcome_counts == {"absorbed": 1}
    assert result.wall_summary == {(TOP_PART_ID, "absorbed", "absorb"): 1}

    events = result.debug["wall_events"]
    assert len(events) == 1
    event = events[0]
    assert event["part_id"] == TOP_PART_ID
    assert event["outcome"] == "absorbed"
    assert event["wall_mode"] == "absorb"
    assert event["hit_y_m"] == pytest.approx(1.0, abs=2.0e-6)
    assert 0.0 < event["hit_time_s"] < 0.5


def test_freeze_keeps_impact_velocity_and_reports_a_distinct_terminal_state(
    tmp_path: Path,
) -> None:
    path = _write_quarter_excursion_case(tmp_path / "freeze")
    boundaries_path = path.parent / "boundaries.csv"
    boundaries = pd.read_csv(boundaries_path)
    boundaries.loc[boundaries["part_id"] == TOP_PART_ID, "wall_law"] = "freeze"
    boundaries.to_csv(boundaries_path, index=False)

    result = simulate(load_case(path))

    event = result.debug["wall_events"][0]
    assert result.state.terminal_state.tolist() == ["frozen"]
    assert result.stats.terminal_counts == {"frozen": 1}
    assert result.stats.wall_outcome_counts == {"frozen": 1}
    assert result.wall_summary == {(TOP_PART_ID, "frozen", "freeze"): 1}
    assert result.state.position_m[0].tolist() == pytest.approx(
        [event["hit_x_m"], event["hit_y_m"]],
        abs=2.0e-6,
    )
    assert result.state.velocity_mps[0].tolist() == pytest.approx(
        [event["v_hit_x_mps"], event["v_hit_y_mps"]],
        abs=1.0e-12,
    )
    assert np.linalg.norm(result.state.velocity_mps[0]) > 0.0

    output_dir = tmp_path / "freeze-output"
    write_result(result, output_dir)
    final = pd.read_csv(output_dir / "final_particles.csv")
    summary = yaml.safe_load((output_dir / "run_summary.json").read_text("utf-8"))
    steps = pd.read_csv(output_dir / "step_summary.csv")
    assert final["final_state"].tolist() == ["frozen"]
    assert summary["frozen_count"] == 1
    assert steps["frozen_count"].tolist() == [1]


def test_outlet_pass_through_exits_at_hit_with_impact_velocity(tmp_path: Path) -> None:
    path = _write_quarter_excursion_case(tmp_path / "outlet")
    boundaries_path = path.parent / "boundaries.csv"
    boundaries = pd.read_csv(boundaries_path)
    top = boundaries["part_id"] == TOP_PART_ID
    boundaries.loc[top, ["role", "wall_law"]] = ["outlet", "pass_through"]
    boundaries.to_csv(boundaries_path, index=False)

    result = simulate(load_case(path))

    event = result.debug["wall_events"][0]
    assert result.state.terminal_state.tolist() == ["escaped"]
    assert result.wall_summary == {(TOP_PART_ID, "escaped", "pass_through"): 1}
    assert result.state.position_m[0].tolist() == pytest.approx(
        [event["hit_x_m"], event["hit_y_m"]],
        abs=2.0e-6,
    )
    assert result.state.velocity_mps[0].tolist() == pytest.approx(
        [event["v_hit_x_mps"], event["v_hit_y_mps"]],
        abs=1.0e-12,
    )


@pytest.mark.parametrize("scale", [1.0e-6, 1.0e3])
def test_public_wall_collision_is_similarity_scale_invariant(
    tmp_path: Path,
    scale: float,
) -> None:
    base_path = _write_quarter_excursion_case(tmp_path / "base")
    scaled_path = _write_quarter_excursion_case(tmp_path / "scaled")
    _scale_quarter_excursion_case(scaled_path, scale)

    base_case = load_case(base_path)
    scaled_case = load_case(scaled_path)
    assert validate_case(base_case).passed
    assert validate_case(scaled_case).passed
    base = simulate(base_case)
    scaled = simulate(scaled_case)

    assert (
        base.state.terminal_state.tolist()
        == scaled.state.terminal_state.tolist()
        == ["absorbed"]
    )
    base_event = base.debug["wall_events"][0]
    scaled_event = scaled.debug["wall_events"][0]
    assert (base_event["part_id"], base_event["outcome"]) == (
        scaled_event["part_id"],
        scaled_event["outcome"],
    )
    assert scaled_event["hit_time_s"] == pytest.approx(
        base_event["hit_time_s"], abs=2.0e-8
    )
    np.testing.assert_allclose(
        scaled.state.position_m / scale,
        base.state.position_m,
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    assert scaled_case.solver_context.plan.boundary.contact_offset_m == pytest.approx(
        scale * base_case.solver_context.plan.boundary.contact_offset_m,
        rel=2.0e-12,
    )


def test_multiple_wall_hits_replay_the_remaining_segment(tmp_path: Path) -> None:
    path = _write_quarter_excursion_case(tmp_path / "multiple-hits")
    particles_path = path.parent / "particles.csv"
    particles = pd.read_csv(particles_path)
    particles.loc[0, ["x_m", "y_m", "vx_mps", "vy_mps"]] = [0.0, 0.0, 8.0, 0.0]
    particles.to_csv(particles_path, index=False)

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["physics"]["forces"] = {}
    config["time"] = {"dt": 0.5, "t_end": 0.5}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    case = load_case(path)
    report = validate_case(case)
    assert report.passed, report.errors

    result = simulate(case)

    assert result.state.terminal_state.tolist() == ["active_free_flight"]
    events = result.debug["wall_events"]
    assert [(event["part_id"], event["outcome"]) for event in events] == [
        (RIGHT_PART_ID, "reflected_specular"),
        (LEFT_PART_ID, "reflected_specular"),
    ]
    assert result.stats.safety_counters["wall_interaction_count"] == 2
    assert result.state.position_m[0, 0] == pytest.approx(0.0, abs=2.0e-5)
    assert result.state.position_m[0, 1] == pytest.approx(0.0, abs=1.0e-14)
    assert result.state.velocity_mps[0].tolist() == pytest.approx(
        [8.0, 0.0], abs=1.0e-13
    )


def test_standard_collision_never_builds_debug_event_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_quarter_excursion_case(tmp_path / "standard-no-debug-rows")
    particles_path = path.parent / "particles.csv"
    particles = pd.read_csv(particles_path)
    particles.loc[0, ["x_m", "y_m", "vx_mps", "vy_mps"]] = [0.0, 0.0, 20.0, 0.0]
    particles.to_csv(particles_path, index=False)

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["physics"]["forces"] = {}
    config["time"] = {"dt": 0.5, "t_end": 0.5}
    config["output"] = {"mode": "standard"}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def fail_row_builder(*_args, **_kwargs):
        raise AssertionError("standard mode evaluated a debug-only event row")

    monkeypatch.setattr(high_fidelity_collision, "_wall_event_row", fail_row_builder)
    monkeypatch.setattr(
        high_fidelity_collision, "_append_max_hit_event", fail_row_builder
    )

    result = simulate(load_case(path))

    assert result.debug == {}
    assert result.stats.safety_counters["wall_interaction_count"] == 5
    assert result.stats.safety_counters["max_hits_reached_count"] == 1


def test_late_specular_hit_advances_the_short_remaining_segment(tmp_path: Path) -> None:
    """A nonterminal hit must not discard the last five percent of a step."""

    path = _write_quarter_excursion_case(tmp_path / "late-hit")
    particles_path = path.parent / "particles.csv"
    particles = pd.read_csv(particles_path)
    particles.loc[0, ["x_m", "y_m", "vx_mps", "vy_mps"]] = [0.04, 0.0, 1.0, 0.0]
    particles.to_csv(particles_path, index=False)

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["physics"]["forces"] = {}
    config["time"] = {"dt": 1.0, "t_end": 1.0}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    case = load_case(path)
    report = validate_case(case)
    assert report.passed, report.errors

    result = simulate(case)

    events = result.debug["wall_events"]
    assert [(event["part_id"], event["outcome"]) for event in events] == [
        (RIGHT_PART_ID, "reflected_specular")
    ]
    assert events[0]["hit_time_s"] == pytest.approx(0.96, abs=3.0e-6)
    assert result.state.velocity_mps[0].tolist() == pytest.approx(
        [-1.0, 0.0], abs=1.0e-13
    )
    # Point-particle reflection gives x(1 s) = 1 - (1 - 0.96) = 0.96.
    # The remaining tolerance is set by hit-time localization; the numerical
    # contact offset itself is geometry-scaled and much smaller for this case.
    assert result.state.position_m[0, 0] == pytest.approx(0.96, abs=3.0e-6)
    assert result.state.position_m[0, 1] == pytest.approx(0.0, abs=1.0e-14)


@pytest.mark.parametrize("time_scale", [1.0e-18, 1.0e6])
def test_wall_hit_localization_has_no_fixed_seconds_floor(
    tmp_path: Path,
    time_scale: float,
) -> None:
    path = _write_quarter_excursion_case(tmp_path / "time-scaled-hit")
    particles_path = path.parent / "particles.csv"
    particles = pd.read_csv(particles_path)
    particles.loc[0, ["x_m", "y_m", "vx_mps", "vy_mps"]] = [
        0.04,
        0.0,
        1.0 / time_scale,
        0.0,
    ]
    particles.to_csv(particles_path, index=False)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["physics"]["forces"] = {}
    config["time"] = {"dt": time_scale, "t_end": time_scale}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = simulate(load_case(path))

    event = result.debug["wall_events"][0]
    assert event["hit_time_s"] / time_scale == pytest.approx(0.96, abs=3.0e-6)
    assert result.state.position_m[0, 0] == pytest.approx(0.96, abs=3.0e-6)
    assert result.state.velocity_mps[0, 0] * time_scale == pytest.approx(
        -1.0, abs=1.0e-12
    )


def _run_field_support_segment(
    *,
    axes: tuple[np.ndarray, np.ndarray],
    field_valid_mask: np.ndarray,
    x0: tuple[float, float],
    velocity: tuple[float, float],
    duration_s: float,
    minimum_substeps: int = 1,
) -> tuple[np.ndarray, int, int, tuple[int, ...]]:
    shape = tuple(axis.size for axis in axes)
    geometry_valid_mask = np.ones(shape, dtype=bool)
    field = regular_field_provider(
        axes,
        field_valid_mask,
        {"ux": np.zeros(shape), "uy": np.zeros(shape)},
    )
    geometry = geometry_provider(
        axes,
        geometry_valid_mask,
        sdf=-np.ones(shape),
        normal_components=(np.zeros(shape), np.ones(shape)),
    )
    compiled = compile_runtime_backend(
        SimpleNamespace(
            geometry_provider=geometry,
            field_provider=field,
            gas=SimpleNamespace(
                density_kgm3=np.nan,
                dynamic_viscosity_Pas=np.nan,
                temperature=np.nan,
            ),
        ),
        spatial_dim=2,
    )
    x_start = np.asarray(x0, dtype=np.float64)
    velocity_vector = np.asarray(velocity, dtype=np.float64)
    trace = trace_motion_segment(
        SegmentMotionRequest(
            position_m=x_start,
            velocity_mps=velocity_vector,
            duration_s=duration_s,
            end_time_s=duration_s,
            spatial_dim=2,
            backend=compiled,
            adaptive_substep_enabled=0,
            adaptive_substep_max_splits=4,
            tau_stokes_s=np.inf,
            particle_diameter_m=1.0e-6,
            particle_density_kgm3=np.nan,
            particle_mass_kg=1.0e-12,
            dep_particle_rel_permittivity=np.nan,
            thermophoretic_coefficient=np.nan,
            body_acceleration_mps2=np.zeros(2),
            gas_density_kgm3=np.nan,
            gas_dynamic_viscosity_Pas=np.nan,
            gas_temperature_K=np.nan,
            gas_molecular_mass_kg=np.nan,
            drag_model_mode=DRAG_MODEL_NONE,
            minimum_substeps=minimum_substeps,
        )
    )
    point_statuses = tuple(
        sample_compiled_valid_mask_status(
            compiled,
            x_start + fraction * duration_s * velocity_vector,
        )
        for fraction in (0.0, 0.25, 0.5, 1.0)
    )
    return (
        trace.endpoint_position_m,
        int(trace.substep_count),
        int(trace.aggregate_support_status),
        point_statuses,
    )


def test_endpoint_only_field_support_exit_is_detected() -> None:
    axis = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    x_end, substeps, status, point_statuses = _run_field_support_segment(
        axes=(axis, axis),
        field_valid_mask=np.ones((3, 3), dtype=bool),
        x0=(0.25, 0.5),
        velocity=(1.0, 0.0),
        duration_s=1.0,
    )

    assert x_end.tolist() == pytest.approx([1.25, 0.5], abs=1.0e-14)
    assert substeps == 1
    assert point_statuses[0] == VALID_MASK_STATUS_CLEAN
    assert point_statuses[2] == VALID_MASK_STATUS_CLEAN
    assert point_statuses[3] == VALID_MASK_STATUS_HARD_INVALID
    assert status == VALID_MASK_STATUS_HARD_INVALID


def test_internal_invalid_island_is_detected_between_clean_endpoint_and_midpoint() -> (
    None
):
    axis = np.linspace(0.0, 1.0, 11)
    field_valid_mask = np.ones((11, 11), dtype=bool)
    field_valid_mask[2:4, :] = False
    x_end, substeps, status, point_statuses = _run_field_support_segment(
        axes=(axis, axis),
        field_valid_mask=field_valid_mask,
        x0=(0.05, 0.5),
        velocity=(1.0, 0.0),
        duration_s=0.8,
        minimum_substeps=2,
    )

    assert x_end.tolist() == pytest.approx([0.85, 0.5], abs=1.0e-14)
    assert substeps == 2
    assert point_statuses[0] == VALID_MASK_STATUS_CLEAN
    assert point_statuses[1] == VALID_MASK_STATUS_HARD_INVALID
    assert point_statuses[2] == VALID_MASK_STATUS_CLEAN
    assert point_statuses[3] == VALID_MASK_STATUS_CLEAN
    assert status == VALID_MASK_STATUS_HARD_INVALID
