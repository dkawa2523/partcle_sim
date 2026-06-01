from __future__ import annotations

from pathlib import Path

import pytest

from particle_tracer_unified.io.comsol_boundary_reader import (
    read_comsol_boundary_map,
    read_comsol_wall_laws,
    validate_wall_law_coverage,
    wall_laws_to_tables,
)


def test_comsol_boundary_reader_builds_wall_tables_with_entity_metadata(tmp_path: Path) -> None:
    boundary_path = tmp_path / "boundary.csv"
    boundary_path.write_text(
        "solver_part_id,comsol_geom_entity_id,selection_name,boundary_type,wall_node,material\n"
        "11,101,inlet,pass_through,pt.wall1,gas\n"
        "12,102,wafer,wall,pt.wall2,sio2\n",
        encoding="utf-8",
    )
    wall_path = tmp_path / "walls.csv"
    wall_path.write_text(
        "solver_part_id,wall_type,stick_probability,restitution_n,restitution_t,diffuse_temperature,material_id\n"
        "11,pass_through,0,1,1,,gas\n"
        "12,stick,1,0,0,,sio2\n",
        encoding="utf-8",
    )

    boundary_rows = read_comsol_boundary_map(boundary_path)
    wall_rows = read_comsol_wall_laws(wall_path)
    validate_wall_law_coverage(boundary_rows, wall_rows)
    materials, walls = wall_laws_to_tables(wall_rows, boundary_rows)

    assert [row.part_id for row in walls.rows] == [11, 12]
    assert walls.rows[0].wall_law == "pass_through"
    assert walls.rows[1].wall_law == "stick"
    assert "comsol_boundary_map" in walls.metadata
    assert len(materials.rows) == 2


def test_comsol_boundary_reader_maps_mixed_specular_probability_to_diffuse_fraction(tmp_path: Path) -> None:
    wall_path = tmp_path / "walls.csv"
    wall_path.write_text(
        "solver_part_id,wall_type,specular_probability,restitution_n,restitution_t,diffuse_temperature,material_id\n"
        "1,mixed_specular_diffuse,0.8,1,1,,mat\n",
        encoding="utf-8",
    )

    _materials, walls = wall_laws_to_tables(read_comsol_wall_laws(wall_path))

    assert walls.rows[0].wall_law == "mixed_specular_diffuse"
    assert walls.rows[0].wall_diffuse_fraction == pytest.approx(0.2)


def test_comsol_boundary_reader_rejects_unsupported_tangential_restitution(tmp_path: Path) -> None:
    wall_path = tmp_path / "walls.csv"
    wall_path.write_text(
        "solver_part_id,wall_type,stick_probability,restitution_n,restitution_t,diffuse_temperature,material_id\n"
        "1,specular,0,1,0.5,,mat\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tangential restitution"):
        read_comsol_wall_laws(wall_path)


def test_comsol_boundary_reader_rejects_unknown_wall_law(tmp_path: Path) -> None:
    wall_path = tmp_path / "walls.csv"
    wall_path.write_text(
        "solver_part_id,wall_type,stick_probability,restitution_n,restitution_t,diffuse_temperature,material_id\n"
        "1,teleport,0,1,1,,mat\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown wall_type"):
        read_comsol_wall_laws(wall_path)


def test_comsol_boundary_reader_rejects_missing_wall_coverage(tmp_path: Path) -> None:
    boundary_path = tmp_path / "boundary.csv"
    boundary_path.write_text(
        "solver_part_id,comsol_geom_entity_id,selection_name,boundary_type,wall_node,material\n"
        "1,10,wall,wall,pt.wall1,mat\n",
        encoding="utf-8",
    )
    wall_path = tmp_path / "walls.csv"
    wall_path.write_text(
        "solver_part_id,wall_type,stick_probability,restitution_n,restitution_t,diffuse_temperature,material_id\n"
        "2,stick,1,0,0,,mat\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing solver_part_id"):
        validate_wall_law_coverage(read_comsol_boundary_map(boundary_path), read_comsol_wall_laws(wall_path))
