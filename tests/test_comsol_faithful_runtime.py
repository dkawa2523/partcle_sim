from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import yaml

from particle_tracer_unified import load_case, simulate, write_result
from particle_tracer_unified.configuration import RunConfig
from particle_tracer_unified.io._comsol_release_projection import (
    apply_release_projection,
)
from particle_tracer_unified.io.comsol import (
    load_comsol_runtime_inputs,
    validate_comsol_runtime_provider,
)
from particle_tracer_unified.io.comsol_manifest import ComsolCaseManifest
from particle_tracer_unified.io.runtime_builder import build_solver_context


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_case(
    root: Path,
    *,
    position: tuple[float, float] = (0.5, 0.5),
    source_part_id: int = 7,
    projection: dict[str, float] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "release.csv").write_text(
        "particle_id,release_time_s,x_m,y_m,vx_mps,vy_mps,mass_kg,drag_diameter_m,"
        "charge_C,source_part_id,density_kgm3,material_id,dep_particle_rel_permittivity,"
        "thermophoretic_coeff\n"
        f"1,0,{position[0]},{position[1]},2,-3,1e-18,1e-6,0,{source_part_id},1000,4,3.9,0.75\n",
        encoding="utf-8",
    )
    (root / "boundaries.csv").write_text(
        "part_id,part_name,comsol_entity_id,role,wall_law,wall_stick_probability,wall_restitution,"
        "wall_diffuse_fraction,wall_critical_sticking_velocity_mps,material_id,material_name,metadata_json\n"
        "7,chamber,70,wall,specular,0,1,0,0,1,steel,{}\n",
        encoding="utf-8",
    )
    axis = np.asarray([0.0, 1.0], dtype=np.float64)
    edges = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    np.savez(
        root / "geometry.npz",
        axis_0=axis,
        axis_1=axis,
        sdf=np.full((2, 2), -1.0),
        valid_mask=np.ones((2, 2), dtype=bool),
        nearest_boundary_part_id_map=np.full((2, 2), 7, dtype=np.int32),
        normal_0=np.zeros((2, 2)),
        normal_1=np.ones((2, 2)),
        boundary_edges=edges,
        boundary_edge_part_ids=np.full(4, 7, dtype=np.int32),
    )
    field = np.zeros((1, 2, 2), dtype=np.float64)
    np.savez(
        root / "field.npz",
        axis_0=axis,
        axis_1=axis,
        times=np.asarray([0.0]),
        valid_mask=np.ones((2, 2), dtype=bool),
        ux=field,
        uy=field,
    )
    formats = {
        "release": ("release.csv", "canonical_particles_csv"),
        "geometry": ("geometry.npz", "precomputed_npz"),
        "field": ("field.npz", "precomputed_npz"),
        "boundaries": ("boundaries.csv", "canonical_boundaries_csv"),
    }
    metadata = {
        "source_solution_number": 1,
        "vacuum_domain_ids": [1],
        "geometry_source": "explicit_comsol_vacuum_domain_selection",
    }
    if projection is not None:
        metadata["release_boundary_projection"] = projection
    manifest = {
        "schema_version": 2,
        "model": {
            "name": "runtime-test",
            "study": "std1",
            "dataset": "dset1",
            "solution": "sol1",
        },
        "coordinates": {
            "coordinate_system": "cartesian_xy",
            "axis_order": ["x", "y"],
            "coordinate_scale_m_per_model_unit": 1.0,
        },
        "time": {"interpolation": "linear", "support_s": [0.0, 0.0]},
        "artifacts": {
            name: {
                "path": filename,
                "format": artifact_format,
                "sha256": _digest(root / filename),
                "size_bytes": (root / filename).stat().st_size,
            }
            for name, (filename, artifact_format) in formats.items()
        },
        "fields": {
            "velocity": {
                "components": {"x": "ux", "y": "uy"},
                "unit": "m/s",
                "scale_to_si": 1.0,
            }
        },
        "forces": [{"solver_force": "drag", "enabled": True, "law": "stokes"}],
        "metadata": metadata,
    }
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")


def _config() -> dict[str, object]:
    return {
        "schema_version": 2,
        "case": {
            "spatial_dim": 2,
            "coordinate_system": "cartesian_xy",
            "adapter": "comsol",
        },
        "inputs": {"comsol_manifest": "manifest.yaml"},
        "physics": {
            "gas": {"dynamic_viscosity_Pas": 1.8e-5},
            "seed": 12345,
        },
        "time": {"dt": 0.1, "t_end": 0.1},
        "output": {"mode": "standard"},
    }


def _load_test_case(root: Path):
    path = root / "run.yaml"
    path.write_text(yaml.safe_dump(_config()), encoding="utf-8")
    return load_case(path)


def _replace_geometry(root: Path, **updates: np.ndarray | None) -> None:
    geometry_path = root / "geometry.npz"
    with np.load(geometry_path) as payload:
        arrays: dict[str, Any] = {
            name: np.asarray(payload[name]) for name in payload.files
        }
    for name, value in updates.items():
        if value is None:
            arrays.pop(name, None)
        else:
            arrays[name] = np.asarray(value)
    np.savez(geometry_path, **arrays)

    manifest_path = root / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    artifact = manifest["artifacts"]["geometry"]
    artifact["sha256"] = _digest(geometry_path)
    artifact["size_bytes"] = geometry_path.stat().st_size
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")


def test_comsol_runtime_uses_only_manifest_artifacts_and_force_inventory(
    tmp_path: Path,
) -> None:
    _write_case(tmp_path)

    runtime = _load_test_case(tmp_path)._context

    assert runtime.particles.position[0].tolist() == pytest.approx([0.5, 0.5])
    assert runtime.particles.velocity[0].tolist() == pytest.approx([2.0, -3.0])
    assert runtime.particles.material_id.tolist() == [4]
    assert runtime.particles.dep_particle_rel_permittivity.tolist() == pytest.approx(
        [3.9]
    )
    assert runtime.particles.thermophoretic_coeff.tolist() == pytest.approx([0.75])
    assert (
        runtime.particles.metadata["release_boundary_projection"]["projected_count"]
        == 0
    )
    assert runtime.plan.drag_model_name == "stokes"
    assert runtime.force_catalog.force_model_name("drag") == "stokes"


def test_comsol_summary_records_resolved_drag_and_manifest_provenance(
    tmp_path: Path,
) -> None:
    _write_case(tmp_path)
    case = _load_test_case(tmp_path)
    output = tmp_path / "result"

    write_result(simulate(case), output)

    summary = yaml.safe_load((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["drag_model"] == "stokes"
    assert summary["execution"]["adapter"] == "comsol"
    provenance = summary["execution"]["provenance"]
    assert provenance["manifest"]["sha256"] == _digest(tmp_path / "manifest.yaml")
    assert provenance["manifest"]["model"] == {
        "name": "runtime-test",
        "study": "std1",
        "dataset": "dset1",
        "solution": "sol1",
    }
    assert provenance["manifest"]["solution_number"] == 1
    assert provenance["manifest"]["geometry"] == {
        "source": "explicit_comsol_vacuum_domain_selection",
        "vacuum_domain_ids": [1],
    }
    assert provenance["manifest"]["artifacts"]["field"]["sha256"] == _digest(
        tmp_path / "field.npz"
    )


def test_wall_release_requires_explicit_manifest_projection(tmp_path: Path) -> None:
    _write_case(tmp_path, position=(0.0, 0.5))

    with pytest.raises(
        ValueError, match=r"declare metadata.release_boundary_projection"
    ):
        _load_test_case(tmp_path)


def test_wall_release_projection_snaps_to_surface_keeping_velocity_and_source(
    tmp_path: Path,
) -> None:
    _write_case(
        tmp_path,
        position=(0.0, 0.5),
        projection={"tolerance_m": 1.0e-8},
    )

    runtime = _load_test_case(tmp_path)._context

    # Snapped onto the declared entity and left there: that is where COMSOL
    # puts an inlet particle, and the solver does not treat the first segment
    # away from it as a hit.
    assert runtime.particles.position[0].tolist() == pytest.approx([0.0, 0.5])
    assert runtime.particles.velocity[0].tolist() == pytest.approx([2.0, -3.0])
    assert runtime.particles.source_part_id.tolist() == [7]
    report = runtime.particles.metadata["release_boundary_projection"]
    assert report["projected_count"] == 1
    assert report["projected_particle_ids"] == [1]


def test_wall_release_projection_rejects_source_provenance_mismatch(
    tmp_path: Path,
) -> None:
    _write_case(
        tmp_path,
        position=(0.0, 0.5),
        source_part_id=8,
        projection={"tolerance_m": 1.0e-8},
    )

    with pytest.raises(ValueError, match="source_part_id=8 is absent"):
        _load_test_case(tmp_path)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"boundary_edge_part_ids": None}, "requires boundary_edges and"),
        (
            {"boundary_edge_part_ids": np.asarray([7, 7, 7], dtype=np.int32)},
            "invalid 2D boundary arrays",
        ),
    ],
)
def test_wall_release_projection_validates_boundary_bundle_before_particles(
    tmp_path: Path,
    updates: dict[str, np.ndarray | None],
    message: str,
) -> None:
    _write_case(tmp_path)
    _replace_geometry(tmp_path, **updates)

    manifest = ComsolCaseManifest.load(tmp_path / "manifest.yaml")
    with pytest.raises(ValueError, match=message):
        load_comsol_runtime_inputs(manifest=manifest, spatial_dim=2)


def test_wall_release_projection_reports_nearest_part_on_provenance_mismatch(
    tmp_path: Path,
) -> None:
    _write_case(
        tmp_path,
        position=(1.0, 0.5),
        source_part_id=7,
        projection={"tolerance_m": 1.0e-8},
    )
    _replace_geometry(
        tmp_path,
        boundary_edge_part_ids=np.asarray([7, 9, 7, 7], dtype=np.int32),
    )

    manifest = ComsolCaseManifest.load(tmp_path / "manifest.yaml")
    with pytest.raises(
        ValueError,
        match=(
            "particle_id=1 boundary provenance mismatch: "
            "source_part_id=7, nearest_part_id=9"
        ),
    ):
        load_comsol_runtime_inputs(manifest=manifest, spatial_dim=2)


def test_comsol_provider_validation_keeps_error_priority_and_time_envelope(
    tmp_path: Path,
) -> None:
    _write_case(tmp_path)
    manifest = ComsolCaseManifest.load(tmp_path / "manifest.yaml")
    missing_with_ghosts = SimpleNamespace(
        field=SimpleNamespace(metadata={"field_ghost_cells": 1}, quantities={})
    )
    with pytest.raises(ValueError, match="without ghost cells"):
        validate_comsol_runtime_provider(manifest, missing_with_ghosts)

    missing = SimpleNamespace(field=SimpleNamespace(metadata={}, quantities={}))
    with pytest.raises(
        ValueError, match=r"missing manifest quantities: \['ux', 'uy'\]"
    ):
        validate_comsol_runtime_provider(manifest, missing)

    quantities = {
        "ux": SimpleNamespace(times=np.asarray([0.0, 1.0], dtype=np.float64)),
        "uy": SimpleNamespace(times=np.asarray([-1.0, 2.0], dtype=np.float64)),
    }
    mismatched = SimpleNamespace(
        field=SimpleNamespace(metadata={}, quantities=quantities)
    )
    with pytest.raises(ValueError, match=r"\(0\.0, 0\.0\) != \(-1\.0, 2\.0\)"):
        validate_comsol_runtime_provider(manifest, mismatched)

    assert validate_comsol_runtime_provider(manifest, None) is None


def test_comsol_provider_validation_accepts_absent_time_samples_or_contract(
    tmp_path: Path,
) -> None:
    _write_case(tmp_path)
    manifest = ComsolCaseManifest.load(tmp_path / "manifest.yaml")
    empty_quantities = {
        name: SimpleNamespace(times=np.asarray([], dtype=np.float64))
        for name in ("ux", "uy")
    }
    empty_provider = SimpleNamespace(
        field=SimpleNamespace(metadata={}, quantities=empty_quantities)
    )
    assert validate_comsol_runtime_provider(manifest, empty_provider) is None

    no_time_contract = replace(manifest, time={})
    populated = {
        name: SimpleNamespace(times=np.asarray([0.0], dtype=np.float64))
        for name in ("ux", "uy")
    }
    provider = SimpleNamespace(field=SimpleNamespace(metadata={}, quantities=populated))
    assert validate_comsol_runtime_provider(no_time_contract, provider) is None


def test_release_projection_rejects_empty_or_degenerate_boundary_inventory(
    tmp_path: Path,
) -> None:
    empty_root = tmp_path / "empty"
    _write_case(empty_root)
    _replace_geometry(
        empty_root,
        boundary_edges=np.empty((0, 2, 2), dtype=np.float64),
        boundary_edge_part_ids=np.empty(0, dtype=np.int32),
    )
    empty_manifest = ComsolCaseManifest.load(empty_root / "manifest.yaml")
    with pytest.raises(ValueError, match="requires closed 2D boundary loops"):
        load_comsol_runtime_inputs(manifest=empty_manifest, spatial_dim=2)

    degenerate_root = tmp_path / "degenerate"
    _write_case(degenerate_root)
    with np.load(degenerate_root / "geometry.npz") as payload:
        edges = np.asarray(payload["boundary_edges"], dtype=np.float64)
    _replace_geometry(
        degenerate_root,
        boundary_edges=np.concatenate([edges, np.asarray([[[2.0, 2.0], [2.0, 2.0]]])]),
        boundary_edge_part_ids=np.asarray([7, 7, 7, 7, 7], dtype=np.int32),
    )
    degenerate_manifest = ComsolCaseManifest.load(degenerate_root / "manifest.yaml")
    with pytest.raises(ValueError, match="non-finite or degenerate boundary segment"):
        load_comsol_runtime_inputs(manifest=degenerate_manifest, spatial_dim=2)


def test_release_projection_rejects_the_obsolete_inward_offset(
    tmp_path: Path,
) -> None:
    """The displaced-release repair is gone, so its key must not be accepted.

    A manifest still declaring it was written for a solver that moved the
    release point off the wall to dodge a spurious self-hit.  That hit no
    longer happens, so silently ignoring the key would leave the author
    believing a displacement is still applied.
    """

    _write_case(tmp_path, position=(0.0, 0.5))
    manifest = ComsolCaseManifest.load(tmp_path / "manifest.yaml")
    manifest = replace(
        manifest,
        metadata={
            **manifest.metadata,
            "release_boundary_projection": {
                "inward_offset_m": 1.0e-6,
                "tolerance_m": 1.0e-8,
            },
        },
    )

    with pytest.raises(ValueError, match="inward_offset_m is obsolete"):
        manifest.validate(strict=True, verify_hashes=False)


def test_release_projection_dimension_policy_preserves_unmodified_table(
    tmp_path: Path,
) -> None:
    _write_case(tmp_path)
    manifest = ComsolCaseManifest.load(tmp_path / "manifest.yaml")
    runtime = load_comsol_runtime_inputs(manifest=manifest, spatial_dim=2)

    same, report = apply_release_projection(
        runtime.particles,
        spatial_dim=3,
        geometry_path=tmp_path / "geometry.npz",
        projection_config=None,
    )
    assert same is runtime.particles
    assert report == {
        "enabled": False,
        "mode": "on_boundary_surface",
        "projected_count": 0,
        "projected_particle_ids": [],
    }
    with pytest.raises(ValueError, match="supported only for 2D"):
        apply_release_projection(
            runtime.particles,
            spatial_dim=3,
            geometry_path=tmp_path / "geometry.npz",
            projection_config={"inward_offset_m": 1.0e-4, "tolerance_m": 1.0e-8},
        )


def test_comsol_runtime_loader_requires_paths_then_coordinate_system(
    tmp_path: Path,
) -> None:
    _write_case(tmp_path)
    manifest = ComsolCaseManifest.load(tmp_path / "manifest.yaml")
    without_release = replace(
        manifest,
        artifacts={
            name: artifact
            for name, artifact in manifest.artifacts.items()
            if name != "release"
        },
    )
    with pytest.raises(ValueError, match="release and boundary paths must be present"):
        load_comsol_runtime_inputs(manifest=without_release, spatial_dim=2)

    without_coordinates = replace(manifest, coordinates={})
    with pytest.raises(ValueError, match="coordinate_system is required"):
        load_comsol_runtime_inputs(manifest=without_coordinates, spatial_dim=2)


def test_solver_context_builder_rejects_untyped_config_before_path_access() -> None:
    with pytest.raises(TypeError, match="requires a typed RunConfig"):
        build_solver_context(
            cast(RunConfig, SimpleNamespace()),
            Path("not-used"),
        )


def test_comsol_manifest_lift_is_rejected_for_rz_no_swirl(tmp_path: Path) -> None:
    _write_case(tmp_path)
    (tmp_path / "release.csv").write_text(
        "particle_id,release_time_s,r_m,z_m,vr_mps,vz_mps,mass_kg,drag_diameter_m,"
        "charge_C,source_part_id,density_kgm3\n"
        "1,0,0.5,0.5,2,-3,1e-18,1e-6,0,7,1000\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["coordinates"]["coordinate_system"] = "axisymmetric_rz"
    manifest["coordinates"]["axis_order"] = ["r", "z"]
    manifest["fields"]["velocity"]["components"] = {"r": "ux", "z": "uy"}
    manifest["forces"].append(
        {
            "solver_force": "lift",
            "enabled": True,
            "model": "saffman",
            "parameters": {"coefficient": 6.46},
        }
    )
    release_artifact = manifest["artifacts"]["release"]
    release_artifact["sha256"] = _digest(tmp_path / "release.csv")
    release_artifact["size_bytes"] = (tmp_path / "release.csv").stat().st_size
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    config = _config()
    config["case"]["coordinate_system"] = "axisymmetric_rz"
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match=r"axisymmetric_rz no-swirl.*lift"):
        load_case(config_path)
