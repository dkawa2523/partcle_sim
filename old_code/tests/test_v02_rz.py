from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from particle_tracer_unified import load_case, simulate, validate_case
from particle_tracer_unified.core.field_sampling import VALID_MASK_STATUS_HARD_INVALID
from particle_tracer_unified.io.canonical_tables import validate_particles_csv
from particle_tracer_unified.solvers import _runtime_valid_mask
from particle_tracer_unified.solvers.field_compilation import compile_runtime_backend
from particle_tracer_unified.solvers.integrator_common import (
    DRAG_MODEL_NONE,
    DRAG_MODEL_STOKES,
)
from particle_tracer_unified.solvers.runtime_execution import prepare_runtime_execution
from particle_tracer_unified.solvers.sampling_backend import CompiledSamplingBackend
from particle_tracer_unified.solvers.segment_motion import (
    SegmentMotionBatchRequest,
    trace_motion_batch,
    trace_motion_segment,
)


def _write_rz_case(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "particle_id": [1],
            "r_m": [0.5],
            "z_m": [0.0],
            "vr_mps": [0.0],
            "vz_mps": [0.0],
            "release_time_s": [0.0],
            "mass_kg": [1.0e-15],
            "drag_diameter_m": [1.0e-6],
            "charge_C": [0.0],
            "source_part_id": [1],
        }
    ).to_csv(root / "particles.csv", index=False)
    part_ids = range(1, 5)
    pd.DataFrame(
        {
            "part_id": part_ids,
            "part_name": [f"boundary_{part_id}" for part_id in part_ids],
            "role": "wall",
            "material_id": 1,
            "material_name": "test",
            "wall_law": "specular",
            "wall_stick_probability": 0.0,
            "wall_restitution": 1.0,
            "wall_diffuse_fraction": 0.0,
            "wall_critical_sticking_velocity_mps": 0.0,
        }
    ).to_csv(root / "boundaries.csv", index=False)
    config = {
        "schema_version": 2,
        "case": {
            "spatial_dim": 2,
            "coordinate_system": "axisymmetric_rz",
            "adapter": "native",
        },
        "inputs": {
            "particles": "particles.csv",
            "boundaries": "boundaries.csv",
            "geometry": {
                "kind": "box",
                "parameters": {
                    "bounds": [0.0, 1.0, -1.0, 1.0],
                    "grid_shape": [11, 11],
                    "boundary_part_ids": [1, 2, 3, 4],
                },
            },
            "field": {
                "kind": "linear_shear",
                "parameters": {"shear_rate": 0.0, "dynamic_viscosity_Pas": 1.8e-5},
            },
        },
        "physics": {"drag": {"model": "none"}, "gas": {}, "forces": {}},
        "time": {"dt": 0.1, "t_end": 0.1},
        "output": {"mode": "standard"},
    }
    path = root / "case.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _scale_rz_case(path: Path, scale: float) -> None:
    particles = pd.read_csv(path.parent / "particles.csv")
    for column in ("r_m", "z_m", "vr_mps", "vz_mps", "drag_diameter_m"):
        particles[column] = particles[column].astype(float) * float(scale)
    particles.to_csv(path.parent / "particles.csv", index=False)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    parameters = config["inputs"]["geometry"]["parameters"]
    parameters["bounds"] = [
        float(value) * float(scale) for value in parameters["bounds"]
    ]
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _exclude_axis_from_geometry(path: Path, radial_min_m: float = 0.1) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    bounds = config["inputs"]["geometry"]["parameters"]["bounds"]
    bounds[0] = float(radial_min_m)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _make_axis_pass_through(path: Path) -> None:
    boundaries = pd.read_csv(path.parent / "boundaries.csv")
    axis = boundaries["part_id"] == 4
    boundaries.loc[axis, ["role", "wall_law"]] = "internal", "pass_through"
    boundaries.to_csv(path.parent / "boundaries.csv", index=False)


def _set_particle(path: Path, **values: float) -> None:
    particles = pd.read_csv(path.parent / "particles.csv")
    for column, value in values.items():
        particles.loc[0, column] = value
    particles.to_csv(path.parent / "particles.csv", index=False)


def _assert_state(result, position, velocity, *, atol: float = 1.0e-12) -> None:
    actual = result.state.position_m[0], result.state.velocity_mps[0]
    np.testing.assert_allclose(actual, (position, velocity), rtol=0.0, atol=atol)


def _rz_motion_request(case, *, drag_model_mode: int) -> SegmentMotionBatchRequest:
    context = case.solver_context
    backend = compile_runtime_backend(context, 2)
    return SegmentMotionBatchRequest(
        position_m=np.array([[0.05, 0.0]]),
        velocity_mps=np.array([[-1.0, 0.0]]),
        active=np.array([True]),
        tau_stokes_s=np.array([np.inf if drag_model_mode == DRAG_MODEL_NONE else 1.0]),
        particle_diameter_m=np.array([1.0e-6]),
        particle_density_kgm3=np.array([1000.0]),
        particle_mass_kg=np.array([1.0e-15]),
        dep_particle_rel_permittivity=np.array([np.nan]),
        thermophoretic_coefficient=np.array([np.nan]),
        end_time_s=0.2,
        duration_s=0.2,
        spatial_dim=2,
        backend=backend,
        body_acceleration_mps2=np.zeros(2),
        gas_density_kgm3=1.2,
        gas_dynamic_viscosity_Pas=1.8e-5,
        gas_temperature_K=300.0,
        gas_molecular_mass_kg=39.948 * 1.66053906660e-27,
        drag_model_mode=int(drag_model_mode),
        adaptive_substep_enabled=0,
        adaptive_substep_max_splits=4,
    )


def test_rz_rejects_undefined_contact_boundary_on_axis(tmp_path: Path) -> None:
    path = _write_rz_case(tmp_path)
    _set_particle(path, r_m=0.05, vr_mps=-1.0)

    case = load_case(path)
    assert validate_case(case).passed
    result = simulate(case)

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["time"] = {"dt": 0.025, "t_end": 0.1}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    refined_dt_result = simulate(load_case(path))

    _assert_state(result, [0.05, 0.0], [1.0, 0.0])
    _assert_state(
        refined_dt_result,
        result.state.position_m[0],
        result.state.velocity_mps[0],
    )
    assert result.wall_summary == {}
    assert result.state.terminal_state.tolist() == ["active_free_flight"]


def test_rz_particle_radius_cannot_be_negative(tmp_path: Path) -> None:
    path = _write_rz_case(tmp_path)
    _set_particle(path, r_m=-1.0e-6)

    with pytest.raises(ValueError, match="r_m must be >= 0"):
        validate_particles_csv(
            path.parent / "particles.csv",
            spatial_dim=2,
            coordinate_system="axisymmetric_rz",
        )

    _set_particle(path, r_m=0.0, z_m=2.0)
    report = validate_case(load_case(path))
    assert not report.passed
    assert "input.initial_geometry" in {issue.code for issue in report.issues}


def test_rz_rejects_pass_through_boundary_on_axis(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = _write_rz_case(tmp_path)
    _set_particle(path, r_m=0.0, vr_mps=-1.0)
    _make_axis_pass_through(path)

    case = load_case(path)
    assert validate_case(case).passed
    execution = prepare_runtime_execution(
        case.solver_context,
        spatial_dim=2,
        plan=case.solver_context.plan,
        debug_buffers=None,
    )
    np.testing.assert_allclose(execution.state.v[0], [1.0, 0.0])
    result = simulate(case)

    assert result.wall_summary == {}
    assert result.state.terminal_state.tolist() == ["active_free_flight"]
    _assert_state(result, [0.1, 0.0], [1.0, 0.0])

    state = execution.state
    state.active[0] = state.released[0] = True
    state.valid_mask_status_flags[0] = VALID_MASK_STATUS_HARD_INVALID
    monkeypatch.setattr(
        _runtime_valid_mask,
        "resolve_valid_mask_retry_then_stop",
        lambda *_args, **_kwargs: SimpleNamespace(
            position=np.array([-0.2, 0.0]),
            velocity=np.array([-0.4, 0.0]),
            accepted_dt=0.05,
            found_valid_prefix=True,
        ),
    )
    outcomes = {}
    _runtime_valid_mask.apply_valid_mask_retry_then_stop(
        execution,
        dt_step=0.1,
        t_end_step=0.1,
        adaptive_substep_enabled=0,
        terminal_outcomes=outcomes,
    )
    np.testing.assert_allclose((state.x[0], state.v[0]), ([0.2, 0.0], [0.4, 0.0]))
    np.testing.assert_allclose(outcomes[0].position, [0.2, 0.0])


def test_rz_no_swirl_ballistic_case_is_deterministic(tmp_path: Path) -> None:
    path = _write_rz_case(tmp_path)
    _exclude_axis_from_geometry(path)
    _set_particle(path, vr_mps=0.1, vz_mps=-0.2)

    case = load_case(path)
    assert validate_case(case).passed
    first = simulate(case)
    second = simulate(case)

    _assert_state(first, [0.51, -0.02], [0.1, -0.2], atol=1.0e-14)
    np.testing.assert_array_equal(first.state.position_m, second.state.position_m)
    np.testing.assert_array_equal(first.state.velocity_mps, second.state.velocity_mps)


def test_rz_scalar_numba_signed_chart(tmp_path: Path, monkeypatch) -> None:
    case = load_case(_write_rz_case(tmp_path))
    request = _rz_motion_request(case, drag_model_mode=DRAG_MODEL_STOKES)
    backend = request.backend
    target_tau_s = float(request.tau_stokes_s[0])
    local_mu_pas = float(request.particle_mass_kg[0]) / (
        3.0 * np.pi * float(request.particle_diameter_m[0]) * target_tau_s
    )
    request = replace(
        request,
        backend=replace(
            backend,
            times=np.asarray([0.0, 0.2]),
            ux=np.stack(
                (
                    np.full_like(backend.ux[0], 0.2),
                    np.full_like(backend.ux[0], 0.4),
                )
            ),
            uy=np.zeros((2, *backend.uy.shape[1:]), dtype=np.float64),
            gas_density=np.repeat(backend.gas_density[:1], 2, axis=0),
            gas_mu=np.full((2, *backend.gas_mu.shape[1:]), local_mu_pas),
            gas_temperature=np.repeat(backend.gas_temperature[:1], 2, axis=0),
        ),
        gas_dynamic_viscosity_Pas=local_mu_pas,
        body_acceleration_mps2=np.asarray([0.3, 0.0]),
        adaptive_substep_enabled=0,
    )

    batch = trace_motion_batch(request)
    sampled_radii: list[float] = []
    original_sample = CompiledSamplingBackend.sample

    def record_sample(self, points_m, time_s, field_request):
        sampled_radii.extend(np.asarray(points_m)[:, 0])
        return original_sample(self, points_m, time_s, field_request)

    monkeypatch.setattr(CompiledSamplingBackend, "sample", record_sample)
    scalar = trace_motion_segment(request.particle_request(0))

    assert min(sampled_radii) >= 0.0
    assert np.min(scalar.coefficient_midpoint_positions_m[:, 0]) >= 0.0
    np.testing.assert_allclose(
        (batch.endpoint_position_m[0], batch.endpoint_velocity_mps[0]),
        (scalar.endpoint_position_m, scalar.endpoint_velocity_mps),
        rtol=2.0e-12,
    )
    assert int(batch.substep_count[0]) == scalar.substep_count == 1
    assert batch.endpoint_position_m[0, 0] < 0.0


@pytest.mark.parametrize(
    "wall_case", [(0.02, 0.1, 4, 0.09, 1.0), (0.0, 1.2, 2, 0.85, -1.0)]
)
def test_rz_material_wall_across_axis(tmp_path: Path, wall_case) -> None:
    radial_min_m, duration_s, part_id, position_r, velocity_r = wall_case
    path = _write_rz_case(tmp_path)
    if radial_min_m:
        _exclude_axis_from_geometry(path, radial_min_m)
    _set_particle(path, r_m=0.05, vr_mps=-1.0)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["time"] = {"dt": duration_s, "t_end": duration_s}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    case = load_case(path)
    assert validate_case(case).passed
    result = simulate(case)

    assert result.wall_summary == {(part_id, "reflected_specular", "specular"): 1}
    _assert_state(result, [position_r, 0.0], [velocity_r, 0.0], atol=2.0e-9)


@pytest.mark.parametrize("scale", [1.0e-9, 1.0e3])
def test_rz_axis_guard_uses_geometry_scale_not_fixed_metres(
    tmp_path: Path,
    scale: float,
) -> None:
    path = _write_rz_case(tmp_path)
    _set_particle(path, vr_mps=0.1, vz_mps=-0.2)
    _make_axis_pass_through(path)
    _scale_rz_case(path, scale)

    case = load_case(path)
    assert validate_case(case).passed
    result = simulate(case)

    assert result.state.terminal_state.tolist() == ["active_free_flight"]
    assert result.wall_summary == {}
    assert case.solver_context.plan.boundary.radial_axis_tolerance_m < 0.5 * scale
