from __future__ import annotations

from pathlib import Path

import pytest

from particle_tracer_unified.io.comsol_release_reader import (
    comsol_release_particles_to_particle_table,
    read_comsol_release_particles,
)


def test_comsol_release_reader_scales_positions_and_builds_particle_table(tmp_path: Path) -> None:
    path = tmp_path / "release.csv"
    path.write_text(
        "particle_id,release_time,x,y,z,vx,vy,vz,mass,diameter,density,charge,source_entity_id,material_id\n"
        "7,1e-9,1,2,3,4,5,6,1e-18,2e-6,1200,-1e-18,12,5\n",
        encoding="utf-8",
    )

    particles = read_comsol_release_particles(
        path,
        coordinate_scale_m_per_model_unit=0.01,
        release_velocity_scale_mps_per_input_unit=0.1,
    )
    table = comsol_release_particles_to_particle_table(particles, spatial_dim=3)

    assert particles[0].x == pytest.approx(0.01)
    assert particles[0].y == pytest.approx(0.02)
    assert particles[0].z == pytest.approx(0.03)
    assert particles[0].vx == pytest.approx(0.4)
    assert particles[0].vy == pytest.approx(0.5)
    assert particles[0].vz == pytest.approx(0.6)
    assert table.count == 1
    assert table.particle_id.tolist() == [7]
    assert table.position[0].tolist() == pytest.approx([0.01, 0.02, 0.03])
    assert table.velocity[0].tolist() == pytest.approx([0.4, 0.5, 0.6])
    assert table.source_part_id.tolist() == [12]
    assert table.material_id.tolist() == [5]


def test_comsol_release_reader_velocity_scale_defaults_to_one(tmp_path: Path) -> None:
    path = tmp_path / "release.csv"
    path.write_text(
        "particle_id,release_time,x,y,vx,vy,mass,diameter,density,charge\n"
        "1,0,0,0,2,3,1e-18,1e-6,1000,0\n",
        encoding="utf-8",
    )

    particles = read_comsol_release_particles(path, coordinate_scale_m_per_model_unit=1.0)

    assert particles[0].vx == pytest.approx(2.0)
    assert particles[0].vy == pytest.approx(3.0)


def test_comsol_release_reader_rejects_invalid_velocity_scale(tmp_path: Path) -> None:
    path = tmp_path / "release.csv"
    path.write_text(
        "particle_id,release_time,x,y,vx,vy,mass,diameter,density,charge\n"
        "1,0,0,0,0,0,1e-18,1e-6,1000,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="release_velocity_scale_mps_per_input_unit"):
        read_comsol_release_particles(
            path,
            coordinate_scale_m_per_model_unit=1.0,
            release_velocity_scale_mps_per_input_unit=0.0,
        )


def test_comsol_release_reader_rejects_missing_required_column(tmp_path: Path) -> None:
    path = tmp_path / "release.csv"
    path.write_text(
        "particle_id,release_time,x,y,vx,vy,mass,diameter,density\n"
        "1,0,0,0,0,0,1e-18,1e-6,1000\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        read_comsol_release_particles(path, coordinate_scale_m_per_model_unit=1.0, strict=True)


def test_comsol_release_reader_rejects_missing_3d_columns_when_requested(tmp_path: Path) -> None:
    path = tmp_path / "release.csv"
    path.write_text(
        "particle_id,release_time,x,y,vx,vy,mass,diameter,density,charge\n"
        "1,0,0,0,0,0,1e-18,1e-6,1000,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns"):
        read_comsol_release_particles(path, coordinate_scale_m_per_model_unit=1.0, spatial_dim=3)


def test_comsol_release_reader_rejects_non_finite_values(tmp_path: Path) -> None:
    path = tmp_path / "release.csv"
    path.write_text(
        "particle_id,release_time,x,y,vx,vy,mass,diameter,density,charge\n"
        "1,0,0,0,0,0,nan,1e-6,1000,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite"):
        read_comsol_release_particles(path, coordinate_scale_m_per_model_unit=1.0)


def test_comsol_release_reader_rejects_duplicate_particle_ids(tmp_path: Path) -> None:
    path = tmp_path / "release.csv"
    path.write_text(
        "particle_id,release_time,x,y,vx,vy,mass,diameter,density,charge\n"
        "1,0,0,0,0,0,1e-18,1e-6,1000,0\n"
        "1,0,0,0,0,0,1e-18,1e-6,1000,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate particle_id"):
        read_comsol_release_particles(path, coordinate_scale_m_per_model_unit=1.0)
