from __future__ import annotations

from pathlib import Path

import pytest

from particle_tracer_unified.io.tables import load_particles_csv


def test_comsol_release_uses_canonical_particle_values_without_loss(
    tmp_path: Path,
) -> None:
    path = tmp_path / "release.csv"
    path.write_text(
        "particle_id,release_time_s,x_m,y_m,z_m,vx_mps,vy_mps,vz_mps,mass_kg,"
        "drag_diameter_m,charge_C,source_part_id,density_kgm3,material_id,"
        "dep_particle_rel_permittivity,thermophoretic_coeff\n"
        "7,1e-9,0.01,0.02,0.03,0.4,0.5,0.6,1e-18,2e-6,-1e-18,12,1200,5,3.9,0.75\n",
        encoding="utf-8",
    )

    table = load_particles_csv(path, 3, "cartesian_xyz")

    assert table.position[0].tolist() == pytest.approx([0.01, 0.02, 0.03])
    assert table.velocity[0].tolist() == pytest.approx([0.4, 0.5, 0.6])
    assert table.mass.tolist() == pytest.approx([1e-18])
    assert table.diameter.tolist() == pytest.approx([2e-6])
    assert table.source_part_id.tolist() == [12]
    assert table.material_id.tolist() == [5]
    assert table.dep_particle_rel_permittivity.tolist() == pytest.approx([3.9])
    assert table.thermophoretic_coeff.tolist() == pytest.approx([0.75])


def test_axisymmetric_release_uses_explicit_rz_columns(tmp_path: Path) -> None:
    path = tmp_path / "release.csv"
    path.write_text(
        "particle_id,release_time_s,r_m,z_m,vr_mps,vz_mps,mass_kg,drag_diameter_m,charge_C,source_part_id\n"
        "1,0,0.1,0.2,2,3,1e-18,1e-6,0,4\n",
        encoding="utf-8",
    )

    table = load_particles_csv(path, 2, "axisymmetric_rz")

    assert table.position[0].tolist() == pytest.approx([0.1, 0.2])
    assert table.velocity[0].tolist() == pytest.approx([2.0, 3.0])


@pytest.mark.parametrize(
    ("header", "row", "message"),
    [
        (
            "particle_id,release_time,x,y,vx,vy,mass,diameter,density,charge",
            "1,0,0,0,0,0,1e-18,1e-6,1000,0",
            "missing required columns",
        ),
        (
            "particle_id,release_time_s,x_m,y_m,vx_mps,vy_mps,mass_kg,drag_diameter_m,charge_C,source_part_id,source_id",
            "1,0,0,0,0,0,1e-18,1e-6,0,1,1",
            "unknown columns",
        ),
        (
            "particle_id,release_time_s,x_m,y_m,vx_mps,vy_mps,mass_kg,drag_diameter_m,charge_C,source_part_id,stick_probability",
            "1,0,0,0,0,0,1e-18,1e-6,0,1,0.5",
            "unknown columns",
        ),
    ],
)
def test_comsol_release_rejects_noncanonical_columns(
    tmp_path: Path,
    header: str,
    row: str,
    message: str,
) -> None:
    path = tmp_path / "release.csv"
    path.write_text(f"{header}\n{row}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_particles_csv(path, 2, "cartesian_xy")


@pytest.mark.parametrize(
    ("extra_header", "values", "coordinate_system", "message"),
    [
        ("", "-1,0,0,0,0,0,1e-18,1e-6,0,1", "cartesian_xy", "particle_id must be >= 0"),
        (
            "",
            "1.5,0,0,0,0,0,1e-18,1e-6,0,1",
            "cartesian_xy",
            "particle_id must contain integers",
        ),
        (
            ",material_id",
            "1,0,0,0,0,0,1e-18,1e-6,0,1,steel",
            "cartesian_xy",
            "material_id must be numeric",
        ),
        (
            ",material_id",
            "1,0,0,0,0,0,1e-18,1e-6,0,1,1.5",
            "cartesian_xy",
            "material_id must contain integers",
        ),
    ],
)
def test_comsol_release_rejects_invalid_particle_and_material_ids(
    tmp_path: Path,
    extra_header: str,
    values: str,
    coordinate_system: str,
    message: str,
) -> None:
    path = tmp_path / "release.csv"
    path.write_text(
        "particle_id,release_time_s,x_m,y_m,vx_mps,vy_mps,mass_kg,drag_diameter_m,charge_C,source_part_id"
        f"{extra_header}\n{values}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_particles_csv(path, 2, coordinate_system)


def test_comsol_release_rejects_negative_axisymmetric_radius(tmp_path: Path) -> None:
    path = tmp_path / "release.csv"
    path.write_text(
        "particle_id,release_time_s,r_m,z_m,vr_mps,vz_mps,mass_kg,drag_diameter_m,charge_C,source_part_id\n"
        "1,0,-0.1,0.2,2,3,1e-18,1e-6,0,4\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="r_m must be >= 0"):
        load_particles_csv(path, 2, "axisymmetric_rz")


def test_comsol_release_rejects_duplicate_particle_ids(tmp_path: Path) -> None:
    path = tmp_path / "release.csv"
    header = (
        "particle_id,release_time_s,x_m,y_m,vx_mps,vy_mps,mass_kg,"
        "drag_diameter_m,charge_C,source_part_id"
    )
    path.write_text(
        f"{header}\n1,0,0,0,0,0,1e-18,1e-6,0,1\n1,0,0,0,0,0,1e-18,1e-6,0,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="particle_id values must be unique"):
        load_particles_csv(path, 2, "cartesian_xy")


@pytest.mark.parametrize(
    ("coordinate_system", "header", "row", "message"),
    [
        (
            "cartesian_xy",
            "x_m,y_m,vx_mps,vy_mps",
            "nan,0,0,0,-1,1e-18,1e-6,0,0",
            "source_part_id must be > 0",
        ),
        (
            "axisymmetric_rz",
            "r_m,z_m,vr_mps,vz_mps",
            "-1,0,0,0,-1,1e-18,1e-6,0,1",
            "r_m must be >= 0",
        ),
        (
            "cartesian_xy",
            "x_m,y_m,vx_mps,vy_mps",
            "0,0,0,0,-1,0,0,0,1",
            "release_time_s must be >= 0",
        ),
    ],
)
def test_particle_validation_preserves_contract_error_order(
    tmp_path: Path,
    coordinate_system: str,
    header: str,
    row: str,
    message: str,
) -> None:
    path = tmp_path / "release.csv"
    path.write_text(
        f"particle_id,{header},release_time_s,mass_kg,drag_diameter_m,charge_C,"
        f"source_part_id\n1,{row}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_particles_csv(path, 2, coordinate_system)
