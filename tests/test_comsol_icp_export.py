from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external" / "comsol_icp_export"
sys.path.insert(0, str(EXTERNAL))

from comsol_icp_export.field_bundle import build_field_bundle_from_table, write_field_bundle  # noqa: E402
from comsol_icp_export.pack_solver_case import (  # noqa: E402
    apply_material_inventory,
    apply_wall_catalog_overrides,
    write_wall_catalog_review,
)
from tools.build_comsol_case import (  # noqa: E402
    FIELD_SUPPORT_BOUNDARY_PART_ID,
    MeshTypeBlock,
    ParsedMesh,
    _all_comsol_edge_entity_reference,
    _field_support_reference_edges,
    _material_and_wall_rows,
    _parse_part_id_list,
)
from tools.enforce_outward_source_velocity import enforce_outward_velocity  # noqa: E402
from particle_tracer_unified.core.datamodel import ParticleTable, QuantitySeriesND, RegularFieldND  # noqa: E402
from particle_tracer_unified.solvers.compiled_field_backend import compile_runtime_backend, sample_compiled_acceleration_vectors  # noqa: E402


def _sample_table() -> pd.DataFrame:
    rows = []
    for r in [0.0, 0.001, 0.002]:
        for z in [0.0, 0.001, 0.002, 0.003]:
            valid = not (r == 0.002 and z == 0.003)
            rows.append(
                {
                    "r": r,
                    "z": z,
                    "valid_mask": 1 if valid else 0,
                    "ux": r + 2.0 * z if valid else np.nan,
                    "uy": 0.5 * r - z if valid else np.nan,
                    "mu": 1.8e-5 + 1.0e-6 * r + 2.0e-6 * z if valid else np.nan,
                    "E_x": 10.0 * r + z if valid else np.nan,
                    "E_y": -3.0 * z + r if valid else np.nan,
                    "T": 320.0 + 10.0 * z if valid else np.nan,
                    "ne": 1.0e16 + 1.0e18 * r if valid else np.nan,
                }
            )
    return pd.DataFrame(rows)


def test_build_field_bundle_from_export_samples(tmp_path: Path) -> None:
    table = _sample_table()
    bundle = build_field_bundle_from_table(
        table,
        q_ref_c=1.602176634e-19,
        m_ref_kg=6.64215627e-26,
        coordinate_scale_m_per_model_unit=0.01,
        coordinate_model_unit="cm",
        metadata={"case_name": "synthetic_icp"},
    )

    assert bundle["axis_0"].shape == (3,)
    assert bundle["axis_1"].shape == (4,)
    np.testing.assert_allclose(bundle["axis_0"], [0.0, 1.0e-5, 2.0e-5])
    np.testing.assert_allclose(bundle["axis_1"], [0.0, 1.0e-5, 2.0e-5, 3.0e-5])
    assert bundle["times"].tolist() == [0.0]
    assert bundle["valid_mask"].shape == (3, 4)
    assert int(np.count_nonzero(bundle["valid_mask"])) == 11
    assert {"ux", "uy", "mu", "E_x", "E_y", "T", "ne"}.issubset(bundle)
    assert "ax" not in bundle
    assert "ay" not in bundle

    mask = bundle["valid_mask"]
    assert np.isnan(bundle["ux"][~mask]).all()

    metadata = json.loads(str(np.asarray(bundle["metadata_json"]).item()))
    assert metadata["case_name"] == "synthetic_icp"
    assert metadata["grid_shape"] == [3, 4]
    assert metadata["valid_node_count"] == 11
    assert metadata["raw_coordinate_model_unit"] == "cm"
    assert metadata["coordinate_scale_m_per_model_unit"] == 0.01
    assert metadata["quantity_summary"]["E_x"]["variation"] > 0.0

    out = tmp_path / "bundle.npz"
    write_field_bundle(bundle, out)
    with np.load(out) as payload:
        assert set(["axis_0", "axis_1", "valid_mask", "ux", "uy", "E_x", "E_y"]).issubset(payload.files)
        assert "ax" not in payload.files
        assert "ay" not in payload.files


def test_solver_uses_particle_q_over_m_for_exported_electric_field() -> None:
    axes = (np.asarray([0.0, 1.0], dtype=np.float64), np.asarray([0.0, 1.0], dtype=np.float64))
    valid_mask = np.ones((2, 2), dtype=bool)
    times = np.asarray([0.0], dtype=np.float64)

    def series(name: str, data) -> QuantitySeriesND:
        return QuantitySeriesND(name=name, unit="", times=times, data=np.asarray(data, dtype=np.float64))

    field = RegularFieldND(
        spatial_dim=2,
        coordinate_system="cartesian_xy",
        axis_names=("x", "y"),
        axes=axes,
        valid_mask=valid_mask,
        quantities={
            "ux": series("ux", np.zeros((1, 2, 2))),
            "uy": series("uy", np.zeros((1, 2, 2))),
            "E_x": series("E_x", np.ones((1, 2, 2)) * 4.0),
            "E_y": series("E_y", np.ones((1, 2, 2)) * -6.0),
        },
    )
    particles = ParticleTable(
        spatial_dim=2,
        particle_id=np.asarray([1], dtype=np.int64),
        position=np.asarray([[0.5, 0.5]], dtype=np.float64),
        velocity=np.asarray([[0.0, 0.0]], dtype=np.float64),
        release_time=np.asarray([0.0], dtype=np.float64),
        mass=np.asarray([2.0], dtype=np.float64),
        diameter=np.asarray([1.0], dtype=np.float64),
        density=np.asarray([1.0], dtype=np.float64),
        charge=np.asarray([-0.5], dtype=np.float64),
        source_part_id=np.asarray([1], dtype=np.int64),
        material_id=np.asarray([1], dtype=np.int64),
        source_event_tag=np.asarray([""], dtype=object),
        source_law_override=np.asarray([""], dtype=object),
        source_speed_scale_override=np.asarray([np.nan], dtype=np.float64),
        stick_probability=np.asarray([0.0], dtype=np.float64),
        dep_particle_rel_permittivity=np.asarray([np.nan], dtype=np.float64),
        thermophoretic_coeff=np.asarray([np.nan], dtype=np.float64),
    )
    runtime = SimpleNamespace(
        geometry_provider=SimpleNamespace(geometry=SimpleNamespace(axes=axes, valid_mask=valid_mask)),
        field_provider=SimpleNamespace(field=field),
        particles=particles,
        gas=SimpleNamespace(density_kgm3=1.0, dynamic_viscosity_Pas=1.8e-5, temperature=300.0),
    )

    backend = compile_runtime_backend(runtime, 2, particles=particles)

    assert backend.acceleration_source == "particle_charge_electric_field"
    assert backend.electric_field_names == ("E_x", "E_y")
    assert backend.electric_q_over_m_Ckg == pytest.approx(0.0)
    accel = sample_compiled_acceleration_vectors(
        backend,
        2,
        0.0,
        np.asarray([[0.5, 0.5]], dtype=np.float64),
        electric_q_over_m=np.asarray([-0.25], dtype=np.float64),
    )
    np.testing.assert_allclose(accel, [[-1.0, 1.5]])


def test_uniform_export_field_is_rejected() -> None:
    table = _sample_table()
    table["ux"] = 1.0
    table["uy"] = 0.0
    table["E_x"] = 5.0
    table["E_y"] = 0.0
    with pytest.raises(ValueError, match="velocity field is spatially uniform"):
        build_field_bundle_from_table(table, q_ref_c=1.0, m_ref_kg=2.0)


def test_nonfinite_required_value_on_valid_support_is_rejected() -> None:
    table = _sample_table()
    table.loc[(table["r"] == 0.001) & (table["z"] == 0.001), "E_x"] = np.nan
    with pytest.raises(ValueError, match="required field E_x is non-finite"):
        build_field_bundle_from_table(table, q_ref_c=1.0, m_ref_kg=2.0, require_nonuniform=False)


def test_incomplete_tensor_grid_is_rejected() -> None:
    table = _sample_table().iloc[:-1].copy()
    with pytest.raises(ValueError, match="complete tensor grid"):
        build_field_bundle_from_table(table, q_ref_c=1.0, m_ref_kg=2.0, require_nonuniform=False)


def test_icp_wall_catalog_uses_half_sticking_for_physical_walls() -> None:
    _materials, walls, _fallback = _material_and_wall_rows([42, FIELD_SUPPORT_BOUNDARY_PART_ID])
    by_part = {int(row["part_id"]): row for row in walls}

    assert by_part[3]["part_name"] == "wafer_3"
    assert by_part[3]["wall_law"] == "specular"
    assert by_part[3]["wall_stick_probability"] == pytest.approx(0.5)
    assert by_part[42]["part_name"] == "sidewall_42"
    assert by_part[42]["wall_stick_probability"] == pytest.approx(0.5)
    assert by_part[FIELD_SUPPORT_BOUNDARY_PART_ID]["part_name"] == "field_support_boundary"
    assert by_part[FIELD_SUPPORT_BOUNDARY_PART_ID]["wall_law"] == "specular"
    assert by_part[FIELD_SUPPORT_BOUNDARY_PART_ID]["wall_stick_probability"] == pytest.approx(0.5)


def test_comsol_edge_entities_are_used_as_field_support_reference() -> None:
    mesh = ParsedMesh(
        sdim=2,
        vertices=np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        type_blocks={
            "tri": MeshTypeBlock("tri", 3, np.asarray([[0, 1, 2]], dtype=np.int64), np.asarray([0], dtype=np.int32)),
            "edg": MeshTypeBlock("edg", 2, np.asarray([[0, 1]], dtype=np.int64), np.asarray([8], dtype=np.int32)),
        },
    )

    entity_edges, entity_ids = _all_comsol_edge_entity_reference(mesh)
    reference_edges, reference_ids = _field_support_reference_edges(
        mesh,
        np.asarray([[[0.0, 0.0], [0.0, 1.0]]], dtype=np.float64),
        np.asarray([FIELD_SUPPORT_BOUNDARY_PART_ID], dtype=np.int32),
    )

    assert entity_edges.shape == (1, 2, 2)
    assert entity_ids.tolist() == [9]
    assert reference_ids.tolist() == [9]
    np.testing.assert_allclose(reference_edges, entity_edges)


def test_material_inventory_updates_generated_entity_maps(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "material_inventory.json").write_text(
        json.dumps(
            {
                "materials": [
                    {"tag": "mat1", "label": "Silicon", "selection_entities": [4]},
                    {"tag": "mat2", "label": "Quartz", "selection_entities": [5]},
                ]
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"comsol_domain_entity_id": 4, "comsol_material_name": "not_exported_from_mphtxt"},
            {"comsol_domain_entity_id": 5, "comsol_material_name": "not_exported_from_mphtxt"},
        ]
    ).to_csv(generated / "comsol_domain_entity_mapping.csv", index=False)
    pd.DataFrame(
        [
            {
                "solver_part_id": 9,
                "adjacent_domain_ids": "4;5",
                "comsol_material_name": "not_exported_from_mphtxt",
            }
        ]
    ).to_csv(generated / "comsol_boundary_entity_mapping.csv", index=False)

    summary = apply_material_inventory(generated)
    domains = pd.read_csv(generated / "comsol_domain_entity_mapping.csv")
    boundaries = pd.read_csv(generated / "comsol_boundary_entity_mapping.csv")

    assert summary["applied"] is True
    assert domains["comsol_material_name"].tolist() == ["Silicon", "Quartz"]
    assert boundaries.loc[0, "comsol_material_name"] == "Silicon|Quartz"


def test_material_inventory_handles_zero_based_selection_entities(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "material_inventory.json").write_text(
        json.dumps({"materials": [{"tag": "mat1", "label": "Silicon", "selection_entities": [3]}]}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"comsol_domain_entity_id": 4, "comsol_material_name": "not_exported_from_mphtxt"}]
    ).to_csv(generated / "comsol_domain_entity_mapping.csv", index=False)
    pd.DataFrame(
        [{"solver_part_id": 9, "adjacent_domain_ids": "4", "comsol_material_name": "not_exported_from_mphtxt"}]
    ).to_csv(generated / "comsol_boundary_entity_mapping.csv", index=False)

    summary = apply_material_inventory(generated)
    domains = pd.read_csv(generated / "comsol_domain_entity_mapping.csv")
    boundaries = pd.read_csv(generated / "comsol_boundary_entity_mapping.csv")

    assert summary["selection_entity_offset"] == 1
    assert domains.loc[0, "comsol_material_name"] == "Silicon"
    assert boundaries.loc[0, "comsol_material_name"] == "Silicon"


def test_wall_catalog_overrides_update_walls_and_materials(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "part_id": 9,
                "part_name": "comsol_wall_9",
                "material_id": 99,
                "material_name": "comsol_wall",
                "wall_law": "specular",
                "wall_restitution": 0.95,
                "wall_diffuse_fraction": 0.0,
                "wall_stick_probability": 0.5,
            }
        ]
    ).to_csv(tmp_path / "part_walls.csv", index=False)
    pd.DataFrame(
        [
            {
                "material_id": 99,
                "material_name": "comsol_wall",
                "source_law": "explicit_csv",
                "source_speed_scale": 1.0,
                "wall_law": "specular",
                "wall_restitution": 0.95,
                "wall_diffuse_fraction": 0.0,
                "wall_stick_probability": 0.5,
            }
        ]
    ).to_csv(tmp_path / "materials.csv", index=False)
    overrides = tmp_path / "wall_catalog_overrides.csv"
    pd.DataFrame(
        [
            {
                "part_id": 9,
                "part_name": "confirmed_rf_liner_9",
                "material_id": 70,
                "material_name": "confirmed_rf_liner",
                "wall_law": "diffuse",
                "wall_restitution": 0.4,
                "wall_diffuse_fraction": 1.0,
                "wall_stick_probability": 0.25,
            }
        ]
    ).to_csv(overrides, index=False)

    summary = apply_wall_catalog_overrides(tmp_path, overrides)
    walls = pd.read_csv(tmp_path / "part_walls.csv")
    materials = pd.read_csv(tmp_path / "materials.csv")

    assert summary["changed_part_ids"] == [9]
    assert walls.loc[0, "part_name"] == "confirmed_rf_liner_9"
    assert walls.loc[0, "wall_law"] == "diffuse"
    assert walls.loc[0, "wall_stick_probability"] == pytest.approx(0.25)
    assert materials.loc[0, "material_id"] == 70
    assert materials.loc[0, "material_name"] == "confirmed_rf_liner"
    assert materials.loc[0, "wall_law"] == "diffuse"


def test_wall_catalog_review_combines_wall_law_and_comsol_mapping(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    pd.DataFrame(
        [
            {
                "part_id": 9,
                "part_name": "comsol_wall_9",
                "material_id": 99,
                "material_name": "comsol_wall",
                "wall_law": "specular",
                "wall_restitution": 0.95,
                "wall_diffuse_fraction": 0.0,
                "wall_stick_probability": 0.5,
            }
        ]
    ).to_csv(tmp_path / "part_walls.csv", index=False)
    pd.DataFrame(
        [
            {
                "solver_part_id": 9,
                "comsol_edge_entity_id": 9,
                "active_in_solver_boundary": True,
                "x_min_m": 0.0,
                "x_max_m": 0.24,
                "y_min_m": 0.13,
                "y_max_m": 0.13,
                "adjacent_domain_ids": "4;5",
                "comsol_material_name": "Silicon|Quartz",
            }
        ]
    ).to_csv(generated / "comsol_boundary_entity_mapping.csv", index=False)

    summary = write_wall_catalog_review(tmp_path, generated)
    review = pd.read_csv(generated / "wall_catalog_review.csv")

    assert summary["written"] is True
    assert review.loc[0, "part_id"] == 9
    assert review.loc[0, "wall_law"] == "specular"
    assert review.loc[0, "comsol_material_name"] == "Silicon|Quartz"


def test_parse_source_part_ids() -> None:
    assert _parse_part_id_list("32,36") == [32, 36]
    assert _parse_part_id_list("32; 36") == [32, 36]
    assert _parse_part_id_list("") is None


def test_enforce_outward_source_velocity_reflects_only_inward_normal_component() -> None:
    df = pd.DataFrame(
        {
            "particle_id": [1, 2],
            "source_x": [0.0, 0.0],
            "source_y": [0.0, 0.0],
            "x": [0.0, 0.0],
            "y": [1.0, 1.0],
            "vx": [3.0, 4.0],
            "vy": [-4.0, 5.0],
        }
    )

    corrected, summary = enforce_outward_velocity(df)

    np.testing.assert_allclose(corrected["vx"], [3.0, 4.0])
    np.testing.assert_allclose(corrected["vy"], [4.0, 5.0])
    np.testing.assert_allclose(corrected["initial_speed_mps"], [5.0, np.sqrt(41.0)])
    assert summary["corrected_particle_count"] == 1
    assert summary["inward_count_before"] == 1
    assert summary["inward_count_after"] == 0
