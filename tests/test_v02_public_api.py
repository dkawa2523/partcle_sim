from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

import particle_tracer_unified as particle_tracer
from particle_tracer_unified.artifacts import validate_artifacts
from particle_tracer_unified.cli import main

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "v02_minimal" / "run_config.yaml"


def test_public_api_runs_without_io_then_writes_only_standard_artifacts(
    tmp_path: Path,
) -> None:
    case = particle_tracer.load_case(EXAMPLE)
    report = particle_tracer.validate_case(case)
    assert report.passed

    result = particle_tracer.simulate(case)
    assert result.stats.particle_count == 2
    assert not hasattr(result, "_runtime_payload")
    assert not (tmp_path / "result").exists()

    manifest = particle_tracer.write_result(result, tmp_path / "result")
    assert sorted(path.name for path in manifest.files.values()) == [
        "final_particles.csv",
        "run_summary.json",
        "wall_summary.csv",
    ]
    assert sorted(path.name for path in (tmp_path / "result").iterdir()) == [
        "final_particles.csv",
        "run_summary.json",
        "wall_summary.csv",
    ]
    final_particles = pd.read_csv(tmp_path / "result" / "final_particles.csv")
    assert "final_state" in final_particles
    assert "terminal_state" not in final_particles
    assert {"x_m", "y_m", "vx_mps", "vy_mps"}.issubset(final_particles.columns)
    summary = json.loads(
        (tmp_path / "result" / "run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["artifact_type"] == "particle_tracer.run_summary"
    assert validate_artifacts(tmp_path / "result")["passed"]


def test_debug_api_uses_the_declared_debug_artifact_contract(tmp_path: Path) -> None:
    value = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    value["inputs"]["particles"] = str((EXAMPLE.parent / "particles.csv").resolve())
    value["inputs"]["boundaries"] = str((EXAMPLE.parent / "boundaries.csv").resolve())
    value["output"] = {"mode": "debug", "trajectory_interval_steps": 1}
    config_path = tmp_path / "debug.yaml"
    config_path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    result = particle_tracer.simulate(particle_tracer.load_case(config_path))
    particle_tracer.write_result(result, tmp_path / "debug-result")

    assert validate_artifacts(tmp_path / "debug-result", require_debug=True)["passed"]


def test_single_cli_exposes_required_subcommands_and_artifact_check(
    tmp_path: Path, capsys
) -> None:
    assert main(["artifacts", str(tmp_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "standard"
    assert sorted(report["failures"]) == [
        "final_particles.csv",
        "run_summary.json",
        "wall_summary.csv",
    ]


def test_writer_refuses_to_mix_v2_results_with_stale_artifacts(tmp_path: Path) -> None:
    result = particle_tracer.simulate(particle_tracer.load_case(EXAMPLE))
    output = tmp_path / "result"
    output.mkdir()
    (output / "solver_report.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the declared artifact contract"):
        particle_tracer.write_result(result, output)


def test_preflight_rejects_missing_gas_quantity_for_enabled_experimental_force(
    tmp_path: Path,
) -> None:
    value = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    value["inputs"]["particles"] = str((EXAMPLE.parent / "particles.csv").resolve())
    value["inputs"]["boundaries"] = str((EXAMPLE.parent / "boundaries.csv").resolve())
    value["physics"]["gas"].pop("density_kgm3")
    value["physics"]["forces"] = {"lift": {"enabled": True}}
    path = tmp_path / "missing-gas.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    report = particle_tracer.validate_case(particle_tracer.load_case(path))

    assert not report.passed
    issue = next(item for item in report.errors if item.code == "physics.gas.missing")
    assert issue.context == {"feature": "lift", "missing": ["density_kgm3"]}


def test_preflight_requires_particle_density_only_for_displaced_fluid_forces(
    tmp_path: Path,
) -> None:
    value = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    particles = pd.read_csv(EXAMPLE.parent / "particles.csv").drop(
        columns=["density_kgm3"]
    )
    particles_path = tmp_path / "particles.csv"
    particles.to_csv(particles_path, index=False)
    value["inputs"]["particles"] = str(particles_path)
    value["inputs"]["boundaries"] = str((EXAMPLE.parent / "boundaries.csv").resolve())
    value["physics"]["forces"] = {"pressure_gradient": {"enabled": True}}
    path = tmp_path / "missing-particle-density.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    report = particle_tracer.validate_case(particle_tracer.load_case(path))

    issue = next(
        item
        for item in report.errors
        if item.code == "physics.particle_density.missing"
    )
    assert issue.context["features"] == ["pressure_gradient"]
    assert issue.context["invalid_count"] == 2


def test_talbot_thermophoresis_requires_explicit_molecular_mass(tmp_path: Path) -> None:
    value = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    value["inputs"]["particles"] = str((EXAMPLE.parent / "particles.csv").resolve())
    value["inputs"]["boundaries"] = str((EXAMPLE.parent / "boundaries.csv").resolve())
    value["physics"]["gas"].pop("molecular_mass_amu")
    value["physics"]["forces"] = {
        "thermophoresis": {
            "enabled": True,
            "model": "talbot",
            "parameters": {
                "gas_thermal_conductivity_W_mK": 0.026,
                "particle_thermal_conductivity_W_mK": 1.4,
            },
        }
    }
    path = tmp_path / "missing-molecular-mass.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    report = particle_tracer.validate_case(particle_tracer.load_case(path))

    issue = next(item for item in report.errors if item.code == "physics.gas.missing")
    assert issue.context == {
        "feature": "thermophoresis",
        "missing": ["molecular_mass_amu"],
    }
    assert any(
        item.code == "physics.force.field.missing"
        and item.context.get("feature") == "thermophoresis"
        for item in report.errors
    )


def test_dep_requires_explicit_particle_permittivity_source(tmp_path: Path) -> None:
    value = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    value["inputs"]["particles"] = str((EXAMPLE.parent / "particles.csv").resolve())
    value["inputs"]["boundaries"] = str((EXAMPLE.parent / "boundaries.csv").resolve())
    value["physics"]["forces"] = {
        "dielectrophoresis": {
            "enabled": True,
            "model": "dc",
            "parameters": {"medium_rel_permittivity": 1.0006},
        }
    }
    path = tmp_path / "missing-dep-particle-property.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    report = particle_tracer.validate_case(particle_tracer.load_case(path))

    issue = next(
        item
        for item in report.errors
        if item.code == "physics.dielectrophoresis.particle_permittivity.missing"
    )
    assert issue.context["invalid_count"] == 2
    assert any(
        item.code == "physics.force.field.missing"
        and item.context.get("feature") == "dielectrophoresis"
        for item in report.errors
    )
