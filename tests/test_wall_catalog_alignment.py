from __future__ import annotations

import csv
from pathlib import Path

from particle_tracer_unified.core.datamodel import WallCatalog, WallPartModel
from particle_tracer_unified.solvers.wall_catalog_alignment import build_wall_catalog_alignment


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_wall_catalog_alignment_compares_current_catalog_with_comsol_mapping(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    _write_rows(
        generated / "comsol_boundary_entity_mapping.csv",
        [
            {
                "solver_part_id": 7,
                "comsol_edge_entity_id": 7,
                "active_in_solver_boundary": "True",
                "adjacent_domain_ids": "3;4",
                "solver_part_name": "glass_wall_7",
                "solver_material_name": "glass_quartz",
                "comsol_material_name": "Glass (quartz)",
            },
            {
                "solver_part_id": 11,
                "comsol_edge_entity_id": 11,
                "active_in_solver_boundary": "False",
                "adjacent_domain_ids": "5;6",
                "solver_part_name": "copper_wall_11",
                "solver_material_name": "copper",
                "comsol_material_name": "Copper",
            },
        ],
    )
    _write_rows(
        generated / "wall_catalog_review.csv",
        [
            {
                "part_id": 7,
                "part_name": "glass_wall_7",
                "material_id": 60,
                "material_name": "glass_quartz",
                "wall_law": "specular",
                "wall_restitution": 0.95,
                "wall_diffuse_fraction": 0.0,
                "wall_stick_probability": 0.5,
            },
        ],
    )
    wall_catalog = WallCatalog(
        default_model=WallPartModel(
            part_id=0,
            part_name="default",
            material_id=0,
            material_name="",
            law_name="specular",
            stick_probability=0.0,
            restitution=1.0,
            diffuse_fraction=0.0,
            critical_sticking_velocity_mps=0.0,
            reflectivity=0.0,
            roughness_rms=0.0,
        ),
        part_models=(
            WallPartModel(
                part_id=7,
                part_name="glass_wall_7",
                material_id=60,
                material_name="glass_quartz",
                law_name="specular",
                stick_probability=0.25,
                restitution=0.95,
                diffuse_fraction=0.0,
                critical_sticking_velocity_mps=0.0,
                reflectivity=0.0,
                roughness_rms=0.0,
            ),
        ),
    )

    summary, rows = build_wall_catalog_alignment(generated_dir=generated, wall_catalog=wall_catalog)

    assert summary["boundary_mapping_part_count"] == 2
    assert summary["review_mismatch_count"] == 1
    assert summary["default_wall_catalog_part_count"] == 1
    assert summary["active_default_wall_catalog_part_count"] == 0
    assert summary["inactive_default_wall_catalog_part_count"] == 1
    assert summary["solver_entity_role_counts"] == {"active_wall": 1, "internal_interface": 1}
    assert summary["review_flags"] == ["wall_catalog_review_mismatch"]
    assert summary["info_flags"] == ["inactive_entities_use_default_wall_catalog"]
    by_part = {int(row["part_id"]): row for row in rows}
    assert by_part[7]["alignment_status"] == "review_mismatch"
    assert by_part[7]["solver_entity_role"] == "active_wall"
    assert by_part[7]["mismatch_fields"] == "wall_stick_probability"
    assert by_part[11]["alignment_status"] == "classified_inactive_default"
    assert by_part[11]["solver_entity_role"] == "internal_interface"
