from __future__ import annotations

from pathlib import Path

import pytest

from particle_tracer_unified.core.datamodel import GasProperties, PreparedRuntime, RuntimeLike
from particle_tracer_unified.core.input_contract import (
    build_initial_particle_field_support_report,
    enforce_initial_particle_field_support,
)
from particle_tracer_unified.io.tables import load_particles_csv


def _prepared_with_particles_csv(path: Path, *, policy: str) -> PreparedRuntime:
    particles = load_particles_csv(path, spatial_dim=2, coordinate_system="cartesian_xy")
    runtime = RuntimeLike(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        particles=particles,
        walls=None,
        materials=None,
        source_events=None,
        process_steps=None,
        compiled_source_events=None,
        geometry_provider=None,
        field_provider=None,
        gas=GasProperties(),
        config_payload={
            "input_contract": {
                "initial_particle_field_support": "off",
                "particle_defaults_policy": policy,
            }
        },
    )
    return PreparedRuntime(runtime=runtime)


def test_particles_csv_records_defaulted_physics_columns(tmp_path: Path) -> None:
    path = tmp_path / "particles.csv"
    path.write_text("particle_id,x,y\n1,0.25,0.5\n", encoding="utf-8")

    particles = load_particles_csv(path, spatial_dim=2, coordinate_system="cartesian_xy")

    assert "mass" in particles.metadata["defaulted_columns"]
    assert "diameter" in particles.metadata["defaulted_columns"]
    assert "release_time" in particles.metadata["defaulted_columns"]
    assert "vx" in particles.metadata["defaulted_columns"]
    assert "vy" in particles.metadata["defaulted_columns"]


def test_particle_defaults_policy_error_fails_input_contract(tmp_path: Path) -> None:
    path = tmp_path / "particles.csv"
    path.write_text("particle_id,x,y\n1,0.25,0.5\n", encoding="utf-8")
    prepared = _prepared_with_particles_csv(path, policy="error")

    report = build_initial_particle_field_support_report(prepared)

    assert report["passed"] is False
    assert report["particle_defaults"]["physics_defaulted_count"] > 0
    with pytest.raises(ValueError, match="Particle defaults are not allowed"):
        enforce_initial_particle_field_support(prepared, tmp_path / "out")


def test_particle_defaults_policy_allow_preserves_legacy_defaults(tmp_path: Path) -> None:
    path = tmp_path / "particles.csv"
    path.write_text("particle_id,x,y\n1,0.25,0.5\n", encoding="utf-8")
    prepared = _prepared_with_particles_csv(path, policy="allow")

    report = enforce_initial_particle_field_support(prepared, tmp_path / "out")

    assert report["passed"] is True
    assert report["particle_defaults"]["physics_defaulted_count"] > 0
