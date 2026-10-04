from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from particle_tracer_unified import load_case, simulate, validate_case, write_result
from particle_tracer_unified.solvers import high_fidelity_runtime


def _write_ballistic_case(root: Path, *, release_time_s: float = 0.75) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "particle_id": 1,
                "x_m": 0.0,
                "y_m": 0.0,
                "vx_mps": 2.0,
                "vy_mps": 0.0,
                "release_time_s": release_time_s,
                "mass_kg": 1.0e-12,
                "drag_diameter_m": 1.0e-6,
                "charge_C": 0.0,
                "source_part_id": 1,
            }
        ]
    ).to_csv(root / "particles.csv", index=False)
    pd.DataFrame(
        [
            {
                "part_id": part_id,
                "part_name": f"wall_{part_id}",
                "role": "wall",
                "material_id": part_id,
                "material_name": f"material_{part_id}",
                "wall_law": "specular",
                "wall_stick_probability": 0.0,
                "wall_restitution": 1.0,
                "wall_diffuse_fraction": 0.0,
                "wall_critical_sticking_velocity_mps": 0.0,
            }
            for part_id in (1, 2, 3, 4)
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
                    "bounds": [-10.0, 10.0, -10.0, 10.0],
                    "grid_shape": [21, 21],
                    "boundary_part_ids": [1, 2, 3, 4],
                },
            },
            "field": {
                "kind": "linear_shear",
                "parameters": {"shear_rate": 0.0, "dynamic_viscosity_Pas": 1.8e-5},
            },
        },
        "physics": {"drag": {"model": "none"}, "gas": {}, "forces": {}, "seed": 1234},
        "time": {"dt": 1.0, "t_end": 1.0},
        "output": {"mode": "standard"},
    }
    path = root / "case.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_public_api_splits_step_at_release_and_has_no_io_in_simulate(
    tmp_path: Path,
) -> None:
    case = load_case(_write_ballistic_case(tmp_path / "case"))
    report = validate_case(case)
    before = set((tmp_path / "case").iterdir())
    result = simulate(case)
    after = set((tmp_path / "case").iterdir())

    assert report.passed
    assert before == after
    assert result.state.position_m[0, 0] == pytest.approx(0.5, abs=1.0e-14)
    assert not hasattr(result, "case")
    assert result.state.position_m.base is None
    assert not result.state.position_m.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        result.state.position_m[0, 0] = 1.0

    artifacts = write_result(result, tmp_path / "output")
    assert sorted(path.name for path in artifacts.files.values()) == [
        "final_particles.csv",
        "run_summary.json",
        "wall_summary.csv",
    ]
    final = pd.read_csv(tmp_path / "output" / "final_particles.csv")
    assert final.loc[0, "x_m"] == pytest.approx(0.5, abs=1.0e-14)
    assert final.loc[0, "final_state"] == "active_free_flight"


def test_release_at_step_end_does_not_move(tmp_path: Path) -> None:
    case = load_case(_write_ballistic_case(tmp_path / "case", release_time_s=1.0))
    result = simulate(case)

    assert result.state.position_m[0, 0] == pytest.approx(0.0, abs=0.0)


def _write_transient_release_grid_case(
    root: Path,
    *,
    include_delayed_particle: bool,
) -> Path:
    path = _write_ballistic_case(root, release_time_s=0.0)
    if include_delayed_particle:
        particles_path = path.parent / "particles.csv"
        particles = pd.read_csv(particles_path)
        delayed = particles.iloc[0].copy()
        delayed["particle_id"] = 2
        delayed["x_m"] = 5.0
        delayed["vx_mps"] = 0.0
        delayed["release_time_s"] = 0.75
        pd.concat([particles, delayed.to_frame().T], ignore_index=True).to_csv(
            particles_path,
            index=False,
        )

    axis = np.linspace(-10.0, 10.0, 21, dtype=np.float64)
    field_shape = (3, axis.size, axis.size)
    ux = np.empty(field_shape, dtype=np.float64)
    ux[0].fill(0.0)
    ux[1].fill(3.0)
    ux[2].fill(-1.0)
    np.savez_compressed(
        path.parent / "field.npz",
        axis_0=axis,
        axis_1=axis,
        times=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        valid_mask=np.ones((axis.size, axis.size), dtype=bool),
        ux=ux,
        uy=np.zeros(field_shape, dtype=np.float64),
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["inputs"]["field"] = {"kind": "precomputed_npz", "path": "field.npz"}
    config["physics"]["drag"] = {"model": "stokes"}
    config["physics"]["gas"] = {"dynamic_viscosity_Pas": 1.8e-5}
    config["time"] = {"dt": 1.0, "t_end": 2.0}
    config["output"] = {"mode": "debug", "trajectory_interval_steps": 1}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_unrelated_delayed_release_preserves_existing_particle_macro_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_trace_motion_batch = high_fidelity_runtime.trace_motion_batch
    existing_particle_intervals: list[tuple[float, float]] = []

    def record_existing_particle_intervals(request, destination):
        if bool(request.active[0]):
            existing_particle_intervals.append(
                (
                    float(request.end_time_s - request.duration_s),
                    float(request.end_time_s),
                )
            )
        return original_trace_motion_batch(request, destination)

    monkeypatch.setattr(
        high_fidelity_runtime,
        "trace_motion_batch",
        record_existing_particle_intervals,
    )
    baseline = simulate(
        load_case(
            _write_transient_release_grid_case(
                tmp_path / "baseline",
                include_delayed_particle=False,
            )
        )
    )
    baseline_intervals = list(existing_particle_intervals)
    existing_particle_intervals.clear()

    with_delayed_release = simulate(
        load_case(
            _write_transient_release_grid_case(
                tmp_path / "delayed",
                include_delayed_particle=True,
            )
        )
    )

    np.testing.assert_array_equal(
        with_delayed_release.state.position_m[0], baseline.state.position_m[0]
    )
    np.testing.assert_array_equal(
        with_delayed_release.state.velocity_mps[0], baseline.state.velocity_mps[0]
    )
    assert baseline_intervals == [(0.0, 1.0), (1.0, 2.0)]
    assert existing_particle_intervals == baseline_intervals
    assert [frame["time_s"] for frame in with_delayed_release.debug["save_frames"]] == [
        0.0,
        1.0,
        2.0,
    ]
    assert [row["time_s"] for row in with_delayed_release.debug["step_summary"]] == [
        1.0,
        2.0,
    ]
    assert with_delayed_release.debug["collision_diagnostics"]["solver_step_count"] == 2


def _set_regular_transient_field(path: Path, times: list[float]) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["inputs"]["field"]["parameters"].update(
        {"time_mode": "transient", "times": list(times)}
    )
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_native_transient_field_must_cover_earliest_integrated_release(
    tmp_path: Path,
) -> None:
    path = _write_ballistic_case(tmp_path / "case", release_time_s=0.25)
    _set_regular_transient_field(path, [0.5, 1.0])

    with pytest.raises(
        ValueError,
        match=r"field_support_s=\[0\.5, 1\.0\].*required_support_s=\[0\.25, 1\.0\]",
    ):
        load_case(path)


def test_release_at_t_end_requires_no_transient_field_interval(tmp_path: Path) -> None:
    path = _write_ballistic_case(tmp_path / "case", release_time_s=1.0)
    _set_regular_transient_field(path, [5.0, 6.0])

    result = simulate(load_case(path))

    assert result.state.position_m[0, 0] == pytest.approx(0.0, abs=0.0)


def test_simulate_rechecks_transient_support_for_manually_replaced_case(
    tmp_path: Path,
) -> None:
    path = _write_ballistic_case(tmp_path / "case", release_time_s=0.0)
    _set_regular_transient_field(path, [0.0, 1.0])
    case = load_case(path)
    invalid = replace(
        case,
        config=replace(case.config, time=replace(case.config.time, t_end=1.1)),
    )

    with pytest.raises(ValueError, match="transient field time support"):
        simulate(invalid)


def test_triangle_transient_field_must_cover_integration_interval(
    tmp_path: Path,
) -> None:
    path = _write_ballistic_case(tmp_path / "case", release_time_s=0.0)
    vertices = np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    times = np.asarray([0.0, 1.0], dtype=np.float64)
    zeros = np.zeros((times.size, vertices.shape[0]), dtype=np.float64)
    np.savez_compressed(
        path.parent / "triangle_field.npz",
        mesh_vertices=vertices,
        mesh_triangles=triangles,
        times=times,
        ux=zeros,
        uy=zeros,
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["inputs"]["field"] = {
        "kind": "precomputed_triangle_mesh_npz",
        "path": "triangle_field.npz",
    }
    config["time"] = {"dt": 0.1, "t_end": 1.1}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"field_support_s=\[0\.0, 1\.0\].*required_support_s=\[0\.0, 1\.1\]",
    ):
        load_case(path)


def test_stokes_response_uses_mass_not_density_end_to_end(tmp_path: Path) -> None:
    path = _write_ballistic_case(tmp_path / "case", release_time_s=0.0)
    particles_path = path.parent / "particles.csv"
    particles = pd.read_csv(particles_path)
    first = particles.iloc[0].to_dict()
    first["density_kgm3"] = 100.0
    second = dict(first)
    second["particle_id"] = 2
    second["density_kgm3"] = 10_000.0
    pd.DataFrame([first, second]).to_csv(particles_path, index=False)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["physics"] = {
        "drag": {"model": "stokes"},
        "gas": {"dynamic_viscosity_Pas": 1.8e-5},
        "forces": {},
        "seed": 1234,
    }
    config["time"] = {"dt": 1.0e-3, "t_end": 1.0e-3}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = simulate(load_case(path))

    np.testing.assert_allclose(
        result.state.position_m[0], result.state.position_m[1], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        result.state.velocity_mps[0], result.state.velocity_mps[1], rtol=0.0, atol=0.0
    )
    tau = 1.0e-12 / (3.0 * np.pi * 1.8e-5 * 1.0e-6)
    decay = np.exp(-1.0e-3 / tau)
    assert result.state.velocity_mps[0, 0] == pytest.approx(2.0 * decay, rel=1.0e-12)
    assert result.state.position_m[0, 0] == pytest.approx(
        2.0 * tau * (1.0 - decay), rel=1.0e-12
    )


def test_constant_body_force_is_exact_ballistic_end_to_end(tmp_path: Path) -> None:
    path = _write_ballistic_case(tmp_path / "case", release_time_s=0.0)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    acceleration = [1.25, -0.75]
    config["physics"]["forces"] = {
        "gravity": {
            "enabled": True,
            "model": "constant_acceleration",
            "parameters": {"acceleration_mps2": acceleration, "buoyancy": False},
        }
    }
    duration = 0.2
    config["time"] = {"dt": duration, "t_end": duration}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = simulate(load_case(path))

    expected_velocity = np.asarray([2.0, 0.0]) + duration * np.asarray(acceleration)
    expected_position = duration * np.asarray(
        [2.0, 0.0]
    ) + 0.5 * duration**2 * np.asarray(acceleration)
    np.testing.assert_allclose(
        result.state.velocity_mps[0], expected_velocity, rtol=1.0e-12, atol=1.0e-14
    )
    np.testing.assert_allclose(
        result.state.position_m[0], expected_position, rtol=1.0e-12, atol=1.0e-14
    )


def _enable_brownian(path: Path, *, dt: float, t_end: float) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["physics"] = {
        "drag": {"model": "stokes"},
        "gas": {
            "temperature_K": 300.0,
            "dynamic_viscosity_Pas": 1.8e-5,
            "density_kgm3": 1.2,
            "molecular_mass_amu": 28.97,
        },
        "forces": {},
        "stochastic": {
            "enabled": True,
            "model": "underdamped_langevin",
            "temperature_source": "gas",
            "seed": 91,
        },
        "seed": 1234,
    }
    config["time"] = {"dt": float(dt), "t_end": float(t_end)}
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_pre_release_interval_does_not_consume_brownian_rng(tmp_path: Path) -> None:
    delayed_path = _write_ballistic_case(tmp_path / "delayed", release_time_s=0.75)
    _enable_brownian(delayed_path, dt=1.0, t_end=1.0)
    immediate_path = _write_ballistic_case(tmp_path / "immediate", release_time_s=0.0)
    _enable_brownian(immediate_path, dt=0.25, t_end=0.25)

    delayed = simulate(load_case(delayed_path))
    immediate = simulate(load_case(immediate_path))

    np.testing.assert_array_equal(delayed.state.position_m, immediate.state.position_m)
    np.testing.assert_array_equal(
        delayed.state.velocity_mps, immediate.state.velocity_mps
    )


def test_unrelated_release_cohort_does_not_change_existing_brownian_path(
    tmp_path: Path,
) -> None:
    baseline_path = _write_ballistic_case(
        tmp_path / "brownian_baseline", release_time_s=0.0
    )
    _enable_brownian(baseline_path, dt=0.25, t_end=1.0)
    expanded_path = _write_ballistic_case(
        tmp_path / "brownian_expanded", release_time_s=0.0
    )
    particles_path = expanded_path.parent / "particles.csv"
    particles = pd.read_csv(particles_path)
    delayed = particles.iloc[0].copy()
    delayed["particle_id"] = 2
    delayed["x_m"] = 5.0
    delayed["vx_mps"] = 0.0
    delayed["release_time_s"] = 0.5
    pd.concat([delayed.to_frame().T, particles], ignore_index=True).to_csv(
        particles_path,
        index=False,
    )
    _enable_brownian(expanded_path, dt=0.25, t_end=1.0)

    baseline = simulate(load_case(baseline_path))
    expanded = simulate(load_case(expanded_path))
    existing = int(np.flatnonzero(expanded.state.particle_id == 1)[0])

    np.testing.assert_array_equal(
        expanded.state.position_m[existing], baseline.state.position_m[0]
    )
    np.testing.assert_array_equal(
        expanded.state.velocity_mps[existing], baseline.state.velocity_mps[0]
    )


def test_preflight_marks_brownian_as_experimental(tmp_path: Path) -> None:
    path = _write_ballistic_case(tmp_path / "case", release_time_s=0.0)
    _enable_brownian(path, dt=0.1, t_end=0.1)

    report = validate_case(load_case(path))

    assert report.passed
    issue = next(
        item for item in report.warnings if item.code == "physics.experimental"
    )
    assert issue.context["features"] == ["brownian_motion"]
