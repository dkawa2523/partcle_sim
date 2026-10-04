from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from particle_tracer_unified.io.canonical_tables import validate_boundaries_csv
from particle_tracer_unified.io.comsol_boundary_reader import (
    CANONICAL_BOUNDARY_COLUMNS,
    ComsolWallLawRow,
    comsol_entity_to_solver_part_id,
    read_comsol_boundaries,
    remap_comsol_boundary_entity_ids,
    validate_geometry_boundary_coverage,
    validate_wall_law_coverage,
    wall_laws_to_boundaries,
)
from particle_tracer_unified.io.tables import load_boundaries_csv


def _write_boundaries(path: Path, rows: list[dict[str, object]]) -> Path:
    columns = [
        *sorted(CANONICAL_BOUNDARY_COLUMNS),
        "selection_name",
        "metadata_json",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def _row(
    part_id: int, *, role: str = "wall", wall_law: str = "specular"
) -> dict[str, object]:
    return {
        "part_id": part_id,
        "part_name": f"part_{part_id}",
        "comsol_entity_id": 100 + part_id,
        "selection_name": f"geom.part_{part_id}",
        "role": role,
        "wall_law": wall_law,
        "wall_stick_probability": 1.0 if wall_law == "stick" else 0.0,
        "wall_restitution": 0.0 if wall_law == "stick" else 1.0,
        "wall_diffuse_fraction": 0.25 if wall_law == "mixed_specular_diffuse" else 0.0,
        "wall_critical_sticking_velocity_mps": 0.0,
        "material_id": 10 + part_id,
        "material_name": f"material {part_id}",
        "metadata_json": json.dumps({"emissivity": 0.8}) if part_id == 2 else "",
    }


def test_canonical_boundaries_build_complete_wall_table(tmp_path: Path) -> None:
    path = _write_boundaries(
        tmp_path / "boundaries.csv",
        [
            _row(1, role="inlet", wall_law="pass_through"),
            _row(2, wall_law="mixed_specular_diffuse"),
        ],
    )

    boundary_rows, wall_rows = read_comsol_boundaries(path)
    validate_wall_law_coverage(boundary_rows, wall_rows, exact=True)
    walls = wall_laws_to_boundaries(wall_rows, boundary_rows)

    assert [row.part_id for row in walls.rows] == [1, 2]
    assert walls.rows[0].role == "inlet"
    assert walls.rows[0].wall_law == "pass_through"
    assert walls.rows[1].wall_law == "mixed_specular_diffuse"
    assert walls.rows[1].wall_diffuse_fraction == pytest.approx(0.25)
    assert walls.rows[1].material_name == "material 2"
    assert walls.rows[1].metadata == {"emissivity": 0.8}
    assert "comsol_boundary_map" in walls.metadata


def test_wall_table_conversion_preserves_defaults_sorting_and_metadata() -> None:
    rows = [
        ComsolWallLawRow(
            solver_part_id=2,
            wall_type="mixed_specular_diffuse",
            stick_probability=0.25,
            restitution_n=0.75,
            mixed_diffuse_fraction=np.nan,
            material_id=12,
            critical_sticking_velocity_mps=0.0,
            material_name="steel",
        ),
        ComsolWallLawRow(
            solver_part_id=1,
            wall_type="cosine_diffuse",
            stick_probability=0.25,
            restitution_n=0.5,
            mixed_diffuse_fraction=0.2,
            material_id=11,
            critical_sticking_velocity_mps=0.0,
            material_name="glass",
            material_metadata={"emissivity": 0.8},
        ),
    ]

    table = wall_laws_to_boundaries(rows)

    assert [row.part_id for row in table.rows] == [1, 2]
    assert table.rows[0].part_name == "comsol_boundary_1"
    assert table.rows[0].role == "wall"
    assert table.rows[0].wall_diffuse_fraction == 1.0
    assert table.rows[1].wall_diffuse_fraction == 0.5
    assert table.metadata == {
        "source": "comsol_wall_laws",
        "material_metadata_by_part": {"1": {"emissivity": 0.8}},
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda rows: rows[0].pop("material_name"),
            "missing canonical boundary columns",
        ),
        (
            lambda rows: rows[0].__setitem__("solver_part_id", 1),
            "unknown canonical boundary columns",
        ),
    ],
)
def test_canonical_boundaries_reject_schema_drift(
    tmp_path: Path, mutate, message: str
) -> None:
    rows = [_row(1)]
    mutate(rows)
    columns = sorted(set(rows[0]))
    path = tmp_path / "boundaries.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match=message):
        read_comsol_boundaries(path)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"role": "mystery"}, "role must be one of"),
        ({"wall_law": "teleport"}, "Unsupported wall_law"),
        ({"wall_stick_probability": 1.1}, "stick_probability must be in"),
        ({"wall_restitution": -0.1}, "restitution must be non-negative"),
        ({"wall_diffuse_fraction": -0.1}, "diffuse_fraction must be in"),
    ],
)
def test_canonical_boundaries_reject_invalid_values(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    row = _row(1)
    row.update(change)
    path = _write_boundaries(tmp_path / "boundaries.csv", [row])

    with pytest.raises(ValueError, match=message):
        read_comsol_boundaries(path)


def test_boundary_row_validation_keeps_identity_before_wall_and_material(
    tmp_path: Path,
) -> None:
    row = _row(1)
    row.update(
        role="mystery",
        wall_law="teleport",
        wall_stick_probability=np.nan,
        material_id=-1,
    )
    path = _write_boundaries(tmp_path / "boundaries.csv", [row])

    with pytest.raises(ValueError, match="role must be one of"):
        read_comsol_boundaries(path)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("part_id", 1.5),
        ("comsol_entity_id", 70.5),
        ("material_id", 2.5),
    ],
)
def test_comsol_boundary_ids_must_be_exact_integers(
    tmp_path: Path,
    column: str,
    value: float,
) -> None:
    row = _row(1)
    row[column] = value
    path = _write_boundaries(tmp_path / "boundaries.csv", [row])

    with pytest.raises(ValueError, match=rf"{column} must contain an integer"):
        read_comsol_boundaries(path)


@pytest.mark.parametrize("value", ["", "steel", -1])
def test_comsol_boundary_requires_explicit_nonnegative_material_id(
    tmp_path: Path,
    value: object,
) -> None:
    row = _row(1)
    row["material_id"] = value
    path = _write_boundaries(tmp_path / "boundaries.csv", [row])

    with pytest.raises(ValueError, match="material_id"):
        read_comsol_boundaries(path)


def test_canonical_boundaries_reject_duplicate_part_id(tmp_path: Path) -> None:
    path = _write_boundaries(tmp_path / "boundaries.csv", [_row(1), _row(1)])

    with pytest.raises(ValueError, match="Duplicate solver_part_id=1"):
        read_comsol_boundaries(path)


def test_canonical_boundaries_reject_duplicate_comsol_entity_id(
    tmp_path: Path,
) -> None:
    first = _row(1)
    second = _row(2)
    second["comsol_entity_id"] = first["comsol_entity_id"]
    path = _write_boundaries(tmp_path / "boundaries.csv", [first, second])

    with pytest.raises(ValueError, match="Duplicate comsol_entity_id=101"):
        read_comsol_boundaries(path)


def test_boundary_identity_maps_comsol_entities_to_solver_parts_for_any_shape(
    tmp_path: Path,
) -> None:
    first = _row(10)
    first["comsol_entity_id"] = 2
    second = _row(20)
    second["comsol_entity_id"] = 1
    path = _write_boundaries(tmp_path / "boundaries.csv", [first, second])
    boundary_rows, _ = read_comsol_boundaries(path)

    identity = comsol_entity_to_solver_part_id(boundary_rows)

    assert identity == {2: 10, 1: 20}
    np.testing.assert_array_equal(
        remap_comsol_boundary_entity_ids(
            np.asarray([1, 2, 1], dtype=np.int64),
            identity,
            context="2D boundary edges",
        ),
        np.asarray([20, 10, 20], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        remap_comsol_boundary_entity_ids(
            np.asarray([[2, 1], [1, 2]], dtype=np.int64),
            identity,
            context="3D boundary triangles",
        ),
        np.asarray([[10, 20], [20, 10]], dtype=np.int32),
    )


def test_boundary_identity_rejects_unmapped_geometry_entity() -> None:
    with pytest.raises(
        ValueError,
        match=r"3D boundary triangles.*missing comsol_entity_id values: \[3\]",
    ):
        remap_comsol_boundary_entity_ids(
            np.asarray([1, 3], dtype=np.int64),
            {1: 20},
            context="3D boundary triangles",
        )


def test_geometry_boundary_coverage_is_exact(tmp_path: Path) -> None:
    path = _write_boundaries(tmp_path / "boundaries.csv", [_row(1), _row(2)])
    boundary_rows, wall_rows = read_comsol_boundaries(path)
    geometry = SimpleNamespace(
        boundary_edge_part_ids=np.asarray([1, 1, 2], dtype=np.int64),
        boundary_triangle_part_ids=None,
    )

    report = validate_geometry_boundary_coverage(geometry, boundary_rows, wall_rows)

    assert report["passed"] is True
    assert report["geometry_part_ids"] == [1, 2]


def test_geometry_boundary_coverage_rejects_stale_or_missing_part(
    tmp_path: Path,
) -> None:
    path = _write_boundaries(tmp_path / "boundaries.csv", [_row(1), _row(3)])
    boundary_rows, wall_rows = read_comsol_boundaries(path)
    geometry = SimpleNamespace(
        boundary_edge_part_ids=np.asarray([1, 2], dtype=np.int64),
        boundary_triangle_part_ids=None,
    )

    with pytest.raises(ValueError, match="must match exactly"):
        validate_geometry_boundary_coverage(geometry, boundary_rows, wall_rows)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"role": "Wall"}, "role must be one of"),
        ({"role": " wall"}, "leading or trailing whitespace"),
        ({"wall_law": "Specular"}, "Unsupported wall_law"),
        ({"wall_law": "specular "}, "leading or trailing whitespace"),
        ({"part_name": " chamber"}, "part_name must not contain"),
        ({"material_name": "steel "}, "material_name must not contain"),
        ({"selection_name": " geom.wall"}, "selection_name must not contain"),
    ],
)
def test_comsol_boundary_text_is_exact_and_never_repaired(
    tmp_path: Path,
    change: dict[str, object],
    message: str,
) -> None:
    row = _row(1)
    row.update(change)
    path = _write_boundaries(tmp_path / "boundaries.csv", [row])

    with pytest.raises(ValueError, match=message):
        read_comsol_boundaries(path)


def _write_native_boundaries(path: Path, row: dict[str, object]) -> Path:
    columns = [
        "part_id",
        "part_name",
        "role",
        "material_id",
        "material_name",
        "wall_law",
        "wall_stick_probability",
        "wall_restitution",
        "wall_diffuse_fraction",
        "wall_critical_sticking_velocity_mps",
    ]
    values = dict(row)
    values["material_id"] = 1
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({column: values[column] for column in columns})
    return path


@pytest.mark.parametrize("column", ["role", "wall_law"])
@pytest.mark.parametrize("decorate", [str.upper, lambda value: f" {value}"])
def test_native_boundary_enums_require_exact_canonical_values(
    tmp_path: Path,
    column: str,
    decorate,
) -> None:
    row = _row(1)
    row[column] = decorate(str(row[column]))
    path = _write_native_boundaries(tmp_path / "boundaries.csv", row)

    with pytest.raises(ValueError, match=rf"boundaries\.{column}"):
        validate_boundaries_csv(path)


def test_native_boundary_loader_preserves_valid_text_without_rewriting(
    tmp_path: Path,
) -> None:
    path = _write_native_boundaries(tmp_path / "boundaries.csv", _row(1))

    walls = load_boundaries_csv(path)

    assert walls.rows[0].role == "wall"
    assert walls.rows[0].wall_law == "specular"
