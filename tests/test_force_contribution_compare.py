from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from particle_tracer_unified.compare.first_step_compare import main as first_step_compare_main
from particle_tracer_unified.solvers.forces import ForceContribution, build_force_catalog
from particle_tracer_unified.solvers.runtime_outputs import _build_force_contribution_rows


def test_force_contribution_row_contains_component_norms() -> None:
    contribution = ForceContribution(
        name="electric",
        acceleration=np.asarray([1.0, -2.0], dtype=float),
        force=np.asarray([3.0, 4.0], dtype=float),
        physical_quantity="acceleration",
        metadata={"comsol_feature": "fpt.force1"},
    )

    row = contribution.as_row()

    assert row["name"] == "electric"
    assert row["accel_norm"] == np.sqrt(5.0)
    assert row["force_norm"] == 5.0
    assert row["metadata_comsol_feature"] == "fpt.force1"


def test_force_contribution_rows_label_added_mass_terms_as_acceleration() -> None:
    catalog = build_force_catalog(
        {
            "solver": {
                "forces": {
                    "pressure_gradient": {"enabled": False},
                    "virtual_mass": {"enabled": False},
                }
            }
        }
    )
    payload = SimpleNamespace(prepared=SimpleNamespace(runtime=SimpleNamespace(force_catalog=catalog)))

    rows = {row["name"]: row for row in _build_force_contribution_rows(payload)}

    assert rows["pressure_gradient"]["physical_quantity"] == "acceleration"
    assert rows["virtual_mass"]["physical_quantity"] == "acceleration"


def test_compare_modules_expose_help() -> None:
    modules = [
        "particle_tracer_unified.compare.field_compare",
        "particle_tracer_unified.compare.acceleration_compare",
        "particle_tracer_unified.compare.trajectory_compare",
        "particle_tracer_unified.compare.boundary_compare",
        "particle_tracer_unified.compare.first_step_compare",
    ]
    for module in modules:
        completed = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0
        assert "usage:" in completed.stdout.lower()


def test_boundary_compare_uses_first_hit_and_writes_summary(tmp_path: Path) -> None:
    python_csv = tmp_path / "python_wall_events.csv"
    comsol_csv = tmp_path / "comsol_wall_events.csv"
    out_csv = tmp_path / "boundary_hit_comparison.csv"
    summary_json = tmp_path / "boundary_hit_comparison.json"
    pd.DataFrame(
        [
            {"particle_id": 1, "hit_time_s": 0.2, "part_id": 20, "outcome": "late", "hit_x_m": 1.0},
            {"particle_id": 1, "hit_time_s": 0.1, "part_id": 10, "outcome": "bounce", "hit_x_m": 0.5},
            {"particle_id": 2, "hit_time_s": 0.3, "part_id": 30, "outcome": "stick", "hit_x_m": 2.0},
        ]
    ).to_csv(python_csv, index=False)
    pd.DataFrame(
        [
            {"particle_id": 1, "hit_time_s": 0.11, "comsol_entity_id": 10, "outcome": "bounce", "hit_x": 0.55},
            {"particle_id": 2, "hit_time_s": 0.31, "comsol_entity_id": 31, "outcome": "freeze", "hit_x": 2.0},
        ]
    ).to_csv(comsol_csv, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "particle_tracer_unified.compare.boundary_compare",
            "--python",
            str(python_csv),
            "--comsol",
            str(comsol_csv),
            "--output",
            str(out_csv),
            "--summary",
            str(summary_json),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    rows = pd.read_csv(out_csv).sort_values("particle_id")
    assert rows.loc[rows["particle_id"] == 1, "hit_time_s_python"].iloc[0] == 0.1
    assert rows.loc[rows["particle_id"] == 1, "part_id_match"].iloc[0]
    assert not rows.loc[rows["particle_id"] == 2, "part_id_match"].iloc[0]
    assert summary_json.exists()


def _write_first_step_case(
    root: Path,
    *,
    coordinate_system: str = "cartesian_xy",
    flow: tuple[float, float] = (0.0, 0.0),
    electric: tuple[float, float] | None = None,
    particle_velocity: tuple[float, float] = (0.0, 0.0),
    particle_mass: float = 1.0,
    particle_charge: float = 0.0,
    particle_diameter: float = 1.0e-3,
    particle_density: float = 1000.0,
    gas_mu: float = 1.0,
    stochastic_enabled: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if coordinate_system == "axisymmetric_rz":
        axis_0 = np.linspace(0.0, 1.0, 5)
        axis_1 = np.linspace(-1.0, 1.0, 5)
        geometry_bounds = [0.0, 1.0, -1.0, 1.0]
        particle_position = (0.5, 0.0)
    else:
        axis_0 = np.linspace(-1.0, 1.0, 5)
        axis_1 = np.linspace(-1.0, 1.0, 5)
        geometry_bounds = [-1.0, 1.0, -1.0, 1.0]
        particle_position = (0.0, 0.0)
    shape = (axis_0.size, axis_1.size)
    payload: dict[str, object] = {
        "axis_0": axis_0,
        "axis_1": axis_1,
        "times": np.asarray([0.0], dtype=np.float64),
        "valid_mask": np.ones(shape, dtype=bool),
        "ux": float(flow[0]) * np.ones(shape, dtype=np.float64),
        "uy": float(flow[1]) * np.ones(shape, dtype=np.float64),
    }
    force_cfg: dict[str, object] = {"electric": {"enabled": False}}
    if electric is not None:
        payload["E_x"] = float(electric[0]) * np.ones(shape, dtype=np.float64)
        payload["E_y"] = float(electric[1]) * np.ones(shape, dtype=np.float64)
        force_cfg["electric"] = {"enabled": True}
    field_path = root / "field.npz"
    np.savez_compressed(field_path, **payload)
    particles_path = root / "particles.csv"
    pd.DataFrame(
        [
            {
                "particle_id": 1,
                "x": float(particle_position[0]),
                "y": float(particle_position[1]),
                "vx": float(particle_velocity[0]),
                "vy": float(particle_velocity[1]),
                "release_time": 0.0,
                "mass": float(particle_mass),
                "diameter": float(particle_diameter),
                "density": float(particle_density),
                "charge": float(particle_charge),
                "source_part_id": 10,
                "material_id": 1,
                "source_event_tag": "",
                "stick_probability": 0.0,
            }
        ]
    ).to_csv(particles_path, index=False)
    solver: dict[str, object] = {
        "dt": 1.0e-5,
        "t_end": 1.0e-3,
        "save_every": 1,
        "integrator": "drag_relaxation",
        "adaptive_substep_enabled": 0,
        "valid_mask_policy": "retry_then_stop",
        "forces": force_cfg,
    }
    if stochastic_enabled:
        solver["stochastic_motion"] = {"enabled": True, "seed": 123, "temperature_source": "gas"}
        force_cfg["brownian"] = {"enabled": True}
    config = {
        "run": {"spatial_dim": 2, "coordinate_system": coordinate_system},
        "paths": {"particles_csv": str(particles_path)},
        "providers": {
            "geometry": {"kind": "box", "bounds": geometry_bounds, "grid_shape": [5, 5]},
            "field": {"kind": "precomputed_npz", "npz_path": str(field_path)},
        },
        "gas": {"temperature_K": 300.0, "dynamic_viscosity_Pas": float(gas_mu), "density_kgm3": 1.0},
        "source": {"preprocess": {"enabled": False}, "default_law": "explicit_csv"},
        "input_contract": {"initial_particle_field_support": "warn"},
        "provider_contract": {"boundary_field_support": "off"},
        "solver": solver,
        "output": {"artifact_mode": "minimal"},
    }
    config_path = root / "run_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_first_step_compare_reports_constant_electric_acceleration(tmp_path: Path) -> None:
    config_path = _write_first_step_case(
        tmp_path / "electric",
        electric=(3.0, -4.0),
        particle_mass=2.0,
        particle_charge=4.0,
    )
    out_dir = tmp_path / "electric_compare"

    rc = first_step_compare_main(["--config", str(config_path), "--output-dir", str(out_dir)])

    assert rc == 0
    forces = pd.read_csv(out_dir / "force_contributions.csv")
    assert forces.loc[0, "source_provenance_group"] == "known_source"
    np.testing.assert_allclose(forces.loc[0, "electric_ax"], 6.0)
    np.testing.assert_allclose(forces.loc[0, "electric_ay"], -8.0)
    np.testing.assert_allclose(forces.loc[0, "total_ax"], 6.0)
    np.testing.assert_allclose(forces.loc[0, "total_ay"], -8.0)
    assert (out_dir / "first_step_error.csv").exists()
    assert (out_dir / "first_step_summary.json").exists()
    assert (out_dir / "first_step_compare_summary.json").exists()
    first_step = pd.read_csv(out_dir / "first_step_error.csv")
    assert first_step.loc[0, "source_provenance_group"] == "known_source"
    assert float(first_step.loc[0, "force_total_update_velocity_residual_mps"]) < 1.0e-9
    assert float(first_step.loc[0, "force_total_update_position_residual_m"]) < 1.0e-12
    assert float(first_step.loc[0, "force_total_euler_velocity_residual_mps"]) < 1.0e-9
    assert float(first_step.loc[0, "force_total_euler_position_residual_m"]) < 1.0e-12


def test_first_step_compare_reports_drag_only_acceleration(tmp_path: Path) -> None:
    config_path = _write_first_step_case(
        tmp_path / "drag",
        flow=(1.0, 0.0),
        particle_velocity=(0.0, 0.0),
        particle_diameter=0.006,
        particle_density=1000.0,
        gas_mu=1.0,
    )
    out_dir = tmp_path / "drag_compare"

    rc = first_step_compare_main(["--config", str(config_path), "--output-dir", str(out_dir)])

    assert rc == 0
    forces = pd.read_csv(out_dir / "force_contributions.csv")
    expected_tau = 1000.0 * 0.006 * 0.006 / (18.0 * 1.0)
    np.testing.assert_allclose(forces.loc[0, "drag_tau_eff_s"], expected_tau)
    np.testing.assert_allclose(forces.loc[0, "drag_ax"], 1.0 / expected_tau)
    np.testing.assert_allclose(forces.loc[0, "drag_ay"], 0.0)
    np.testing.assert_allclose(forces.loc[0, "electric_ax"], 0.0)
    np.testing.assert_allclose(forces.loc[0, "total_ax"], 1.0 / expected_tau)
    first_step = pd.read_csv(out_dir / "first_step_error.csv")
    assert float(first_step.loc[0, "force_total_update_velocity_residual_mps"]) < 1.0e-12
    assert float(first_step.loc[0, "force_total_update_position_residual_m"]) < 1.0e-14
    assert float(first_step.loc[0, "force_total_euler_velocity_residual_mps"]) > 0.0


def test_first_step_compare_disables_stochastic_by_default(tmp_path: Path) -> None:
    config_path = _write_first_step_case(
        tmp_path / "stochastic",
        flow=(0.0, 0.0),
        stochastic_enabled=True,
    )
    out_a = tmp_path / "stochastic_compare_a"
    out_b = tmp_path / "stochastic_compare_b"

    first_step_compare_main(["--config", str(config_path), "--output-dir", str(out_a), "--seed", "7"])
    first_step_compare_main(["--config", str(config_path), "--output-dir", str(out_b), "--seed", "7"])

    summary = json.loads((out_a / "first_step_compare_summary.json").read_text(encoding="utf-8"))
    forces = pd.read_csv(out_a / "force_contributions.csv")
    first_a = pd.read_csv(out_a / "first_step_error.csv")
    first_b = pd.read_csv(out_b / "first_step_error.csv")
    assert summary["stochastic_policy"] == "off"
    assert int(summary["stochastic_disabled_for_compare"]) == 1
    np.testing.assert_allclose(forces.loc[0, "brownian_ax"], 0.0)
    np.testing.assert_allclose(forces.loc[0, "brownian_ay"], 0.0)
    pd.testing.assert_frame_equal(first_a, first_b)


def test_first_step_compare_controls_stochastic_from_config_with_seed(tmp_path: Path) -> None:
    config_path = _write_first_step_case(
        tmp_path / "stochastic_controlled",
        flow=(0.0, 0.0),
        stochastic_enabled=True,
    )
    out_a = tmp_path / "stochastic_controlled_a"
    out_b = tmp_path / "stochastic_controlled_b"

    first_step_compare_main(
        ["--config", str(config_path), "--output-dir", str(out_a), "--stochastic", "from-config", "--seed", "7"]
    )
    first_step_compare_main(
        ["--config", str(config_path), "--output-dir", str(out_b), "--stochastic", "from-config", "--seed", "7"]
    )

    summary = json.loads((out_a / "first_step_summary.json").read_text(encoding="utf-8"))
    first_a = pd.read_csv(out_a / "first_step_error.csv")
    first_b = pd.read_csv(out_b / "first_step_error.csv")
    assert summary["stochastic_policy"] == "from-config"
    assert int(summary["stochastic_disabled_for_compare"]) == 0
    assert int(summary["stochastic_controlled_by_seed"]) == 1
    pd.testing.assert_frame_equal(first_a, first_b)


def test_first_step_compare_dt_sweep_reports_drag_convergence(tmp_path: Path) -> None:
    config_path = _write_first_step_case(
        tmp_path / "drag_dt_sweep",
        flow=(1.0, 0.0),
        particle_velocity=(0.0, 0.0),
        particle_diameter=0.006,
        particle_density=1000.0,
        gas_mu=1.0,
    )
    out_dir = tmp_path / "drag_dt_sweep_compare"

    rc = first_step_compare_main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(out_dir),
            "--dt-sweep",
            "4e-5,2e-5,1e-5",
        ]
    )

    assert rc == 0
    summary_path = out_dir / "dt_sweep_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    runs = summary["runs"]
    update_residuals = [
        float(run["force_update_velocity_residual_mps"]["max"])
        for run in runs
    ]
    euler_residuals = [
        float(run["force_euler_velocity_residual_mps"]["max"])
        for run in runs
    ]
    assert max(update_residuals) < 1.0e-12
    assert euler_residuals[2] < euler_residuals[1] < euler_residuals[0]
    assert runs[1]["force_euler_velocity_residual_max_ratio_vs_previous"] < 1.0
    assert runs[2]["force_euler_velocity_residual_max_ratio_vs_previous"] < 1.0


def test_first_step_compare_merges_reference_errors(tmp_path: Path) -> None:
    config_path = _write_first_step_case(tmp_path / "reference", electric=(1.0, 0.0), particle_mass=1.0, particle_charge=1.0)
    reference_path = tmp_path / "reference.csv"
    pd.DataFrame(
        [
            {
                "particle_id": 1,
                "x1_ref": 0.0,
                "y1_ref": 0.0,
                "vx1_ref": 0.0,
                "vy1_ref": 0.0,
            }
        ]
    ).to_csv(reference_path, index=False)
    out_dir = tmp_path / "reference_compare"

    rc = first_step_compare_main(
        ["--config", str(config_path), "--reference", str(reference_path), "--output-dir", str(out_dir)]
    )

    assert rc == 0
    first_step = pd.read_csv(out_dir / "first_step_error.csv")
    assert np.isfinite(float(first_step.loc[0, "position_error_m"]))
    assert np.isfinite(float(first_step.loc[0, "velocity_error_mps"]))


def test_first_step_compare_axisymmetric_rz_uses_rz_columns(tmp_path: Path) -> None:
    config_path = _write_first_step_case(
        tmp_path / "axisymmetric",
        coordinate_system="axisymmetric_rz",
        flow=(0.0, 0.0),
    )
    out_dir = tmp_path / "axisymmetric_compare"

    rc = first_step_compare_main(["--config", str(config_path), "--output-dir", str(out_dir)])

    assert rc == 0
    summary = json.loads((out_dir / "first_step_compare_summary.json").read_text(encoding="utf-8"))
    forces = pd.read_csv(out_dir / "force_contributions.csv")
    first_step = pd.read_csv(out_dir / "first_step_error.csv")
    assert summary["coordinate_system"] == "axisymmetric_rz"
    assert summary["axis_names"] == ["r", "z"]
    assert {"r", "z", "drag_ar", "drag_az"}.issubset(set(forces.columns))
    assert {"r0", "z0", "r1_solver", "z1_solver", "vr1_solver", "vz1_solver"}.issubset(set(first_step.columns))
    assert "x0" not in first_step.columns
