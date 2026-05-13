from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from particle_tracer_unified.io.runtime_builder import build_runtime_from_config, prepare_runtime


def _write_runtime_npz(root: Path, *, field_ghost_cells: int = 0) -> None:
    axes = (
        np.asarray([0.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 1.0], dtype=np.float64),
    )
    shape = (2, 2)
    valid_mask = np.ones(shape, dtype=bool)
    np.savez(
        root / "geometry.npz",
        axis_0=axes[0],
        axis_1=axes[1],
        sdf=np.ones(shape, dtype=np.float64),
        valid_mask=valid_mask,
        nearest_boundary_part_id_map=np.ones(shape, dtype=np.int32),
        normal_0=np.zeros(shape, dtype=np.float64),
        normal_1=np.ones(shape, dtype=np.float64),
        boundary_edges=np.asarray(
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 1.0]],
                [[1.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        boundary_edge_part_ids=np.asarray([1, 1, 1, 1], dtype=np.int32),
    )
    metadata = {"field_ghost_cells": int(field_ghost_cells)} if field_ghost_cells else {}
    field_values = np.zeros((1, 2, 2), dtype=np.float64)
    np.savez(
        root / "field.npz",
        axis_0=axes[0],
        axis_1=axes[1],
        times=np.asarray([0.0], dtype=np.float64),
        valid_mask=valid_mask,
        ux=field_values,
        uy=field_values,
        E_x=np.ones((1, 2, 2), dtype=np.float64),
        E_y=np.zeros((1, 2, 2), dtype=np.float64),
        metadata_json=json.dumps(metadata),
    )


def _write_comsol_tables(root: Path) -> None:
    (root / "release.csv").write_text(
        "particle_id,release_time,x,y,vx,vy,mass,diameter,density,charge,source_entity_id,material_id\n"
        "1,0,0.5,0.5,0,0,1e-18,1e-6,1000,0,1,1\n",
        encoding="utf-8",
    )
    (root / "boundary.csv").write_text(
        "solver_part_id,comsol_geom_entity_id,selection_name,boundary_type,wall_node,material\n"
        "1,10,wall,wall,pt.wall1,mat\n",
        encoding="utf-8",
    )
    (root / "walls.csv").write_text(
        "solver_part_id,wall_type,stick_probability,restitution_n,restitution_t,diffuse_temperature,material_id\n"
        "1,stick,1,0,0,,1\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "model": {"study": "std1", "dataset": "dset1", "solution": "sol1"},
        "coordinates": {
            "coordinate_system": "cartesian_xy",
            "coordinate_scale_m_per_model_unit": 1.0,
        },
        "fields": [
            {"name": "u", "physical_quantity": "velocity", "components": {"x": "ux", "y": "uy"}},
            {"name": "E", "physical_quantity": "electric_field", "components": {"x": "E_x", "y": "E_y"}},
        ],
        "particles": {"release_table": "release.csv"},
        "boundaries": {"map_file": "boundary.csv", "wall_law_file": "walls.csv"},
        "forces": [
            {"solver_force": "drag", "enabled": True, "law": "stokes_cunningham", "physical_quantity": "force"},
            {"solver_force": "electric", "enabled": True, "law": "particle_charge", "physical_quantity": "acceleration"},
        ],
    }
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")


def _config() -> dict[str, object]:
    return {
        "mode": "comsol_faithful",
        "comsol": {"manifest": "manifest.yaml"},
        "run": {"spatial_dim": 2},
        "paths": {},
        "providers": {
            "geometry": {"kind": "precomputed_npz", "npz_path": "geometry.npz"},
            "field": {"kind": "precomputed_npz", "npz_path": "field.npz"},
        },
        "gas": {"temperature_K": 300.0, "dynamic_viscosity_Pas": 1.8e-5, "density_kgm3": 1.0},
        "source": {"preprocess": {"enabled": False}},
        "solver": {"dt": 1.0e-8, "t_end": 1.0e-7, "integrator": "etd2"},
    }


def test_comsol_faithful_runtime_loads_manifest_release_and_wall_tables(tmp_path: Path) -> None:
    _write_runtime_npz(tmp_path)
    _write_comsol_tables(tmp_path)

    runtime = build_runtime_from_config(_config(), tmp_path)
    prepared = prepare_runtime(runtime)

    assert prepared.source_preprocess is None
    assert runtime.particles.count == 1
    assert runtime.particles.position[0].tolist() == pytest.approx([0.5, 0.5])
    assert runtime.particles.metadata["source"] == "comsol_release_table"
    assert runtime.particles.metadata["coordinate_scale_m_per_model_unit"] == pytest.approx(1.0)
    assert runtime.particles.metadata["release_velocity_scale_mps_per_input_unit"] == pytest.approx(1.0)
    assert runtime.config_payload["solver"]["valid_mask_policy"] == "strict_clean"
    assert runtime.config_payload["solver"]["drag_model"] == "stokes_cunningham"
    assert runtime.force_catalog.model("drag") == "stokes_cunningham"
    assert runtime.walls.metadata["comsol_boundary_map"][0]["comsol_geom_entity_id"] == 10


def test_comsol_faithful_runtime_manifest_disables_unlisted_electric_force(tmp_path: Path) -> None:
    _write_runtime_npz(tmp_path)
    _write_comsol_tables(tmp_path)
    manifest_path = tmp_path / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["forces"] = [
        {"solver_force": "drag", "enabled": True, "law": "stokes", "physical_quantity": "force"}
    ]
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    runtime = build_runtime_from_config(_config(), tmp_path)

    assert runtime.force_catalog.enabled("electric") is False


def test_comsol_faithful_runtime_rejects_source_preprocess(tmp_path: Path) -> None:
    _write_runtime_npz(tmp_path)
    _write_comsol_tables(tmp_path)
    config = _config()
    config["source"] = {"preprocess": {"enabled": True}}

    with pytest.raises(ValueError, match="source.preprocess.enabled"):
        build_runtime_from_config(config, tmp_path)


def test_comsol_faithful_runtime_rejects_mixed_policy_warning(tmp_path: Path) -> None:
    _write_runtime_npz(tmp_path)
    _write_comsol_tables(tmp_path)
    config = _config()
    config["field_support"] = {"mixed_stencil_policy": "warn"}

    with pytest.raises(ValueError, match="mixed_stencil_policy"):
        build_runtime_from_config(config, tmp_path)


def test_comsol_faithful_runtime_rejects_non_strict_valid_mask_policy(tmp_path: Path) -> None:
    _write_runtime_npz(tmp_path)
    _write_comsol_tables(tmp_path)
    config = _config()
    config["solver"] = {**config["solver"], "valid_mask_policy": "retry_then_stop"}

    with pytest.raises(ValueError, match="valid_mask_policy"):
        build_runtime_from_config(config, tmp_path)


def test_comsol_faithful_runtime_rejects_field_ghost_cells(tmp_path: Path) -> None:
    _write_runtime_npz(tmp_path, field_ghost_cells=8)
    _write_comsol_tables(tmp_path)

    with pytest.raises(ValueError, match="without ghost cells"):
        build_runtime_from_config(_config(), tmp_path)
