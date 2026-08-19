from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from particle_tracer_unified import load_case, validate_case
from particle_tracer_unified.force_models import ForceModel
from particle_tracer_unified.io.comsol_manifest import ComsolCaseManifest


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_artifact_metadata(
    manifest_path: Path,
    *artifact_names: str,
) -> None:
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    for name in artifact_names:
        artifact = payload["artifacts"][name]
        artifact_path = manifest_path.parent / artifact["path"]
        artifact["sha256"] = _digest(artifact_path)
        artifact["size_bytes"] = artifact_path.stat().st_size
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _rewrite_npz(path: Path, *, excluded: set[str], **updates: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as source:
        arrays = {
            name: np.asarray(source[name])
            for name in source.files
            if name not in excluded
        }
    archive: dict[str, Any] = {**arrays, **updates}
    np.savez(path, **archive)


def _write_v2_bundle(root: Path, *, boundary_part_id: int = 7) -> Path:
    release = root / "particles.csv"
    release.write_text(
        "particle_id,release_time_s,x_m,y_m,vx_mps,vy_mps,mass_kg,drag_diameter_m,charge_C,"
        "source_part_id,density_kgm3\n"
        "1,0,0.005,0.005,0,0,5.235987755982989e-16,1e-6,0,7,1000\n",
        encoding="utf-8",
    )
    boundaries = root / "boundaries.csv"
    boundaries.write_text(
        "part_id,part_name,comsol_entity_id,role,wall_law,wall_stick_probability,wall_restitution,"
        "wall_diffuse_fraction,wall_critical_sticking_velocity_mps,material_id,material_name,metadata_json\n"
        f"{boundary_part_id},chamber,70,wall,stick,1,0,0,0,1,steel,{{}}\n",
        encoding="utf-8",
    )
    axes = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    shape = (3, 3)
    np.savez(
        root / "geometry.npz",
        axis_0=axes,
        axis_1=axes,
        sdf=np.ones(shape, dtype=np.float64),
        valid_mask=np.ones(shape, dtype=bool),
        nearest_boundary_part_id_map=np.full(shape, 7, dtype=np.int32),
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
        boundary_edge_part_ids=np.full(4, 7, dtype=np.int32),
    )
    field_values = np.full((2, *shape), 20.0, dtype=np.float64)
    np.savez(
        root / "field.npz",
        axis_0=axes,
        axis_1=axes,
        times=np.asarray([0.0, 1.0], dtype=np.float64),
        valid_mask=np.ones(shape, dtype=bool),
        raw_component_0=np.zeros_like(field_values),
        raw_component_1=field_values,
        raw_electric_0=np.full_like(field_values, 2.0),
        raw_electric_1=np.full_like(field_values, -3.0),
        raw_temperature=np.full_like(field_values, 300.0),
    )
    artifacts = {
        "release": ("particles.csv", "canonical_particles_csv"),
        "geometry": ("geometry.npz", "precomputed_npz"),
        "field": ("field.npz", "precomputed_npz"),
        "boundaries": ("boundaries.csv", "canonical_boundaries_csv"),
    }
    manifest = {
        "schema_version": 2,
        "model": {
            "name": "unit-test",
            "study": "std1",
            "dataset": "dset1",
            "solution": "sol1",
        },
        "coordinates": {
            "coordinate_system": "cartesian_xy",
            "axis_order": ["x", "y"],
            "coordinate_scale_m_per_model_unit": 0.01,
        },
        "time": {"interpolation": "linear", "support_s": [0.0, 1.0]},
        "artifacts": {
            name: {
                "path": filename,
                "format": artifact_format,
                "sha256": _digest(root / filename),
                "size_bytes": (root / filename).stat().st_size,
            }
            for name, (filename, artifact_format) in artifacts.items()
        },
        "fields": {
            "velocity": {
                "unit": "m/s",
                "scale_to_si": 0.1,
                "components": {"x": "raw_component_1", "y": "raw_component_0"},
            }
        },
        "forces": [{"solver_force": "drag", "enabled": True, "law": "stokes"}],
        "metadata": {
            "source_solution_number": 1,
            "vacuum_domain_ids": [1],
            "geometry_source": "explicit_comsol_vacuum_domain_selection",
        },
    }
    path = root / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def _runtime_config() -> dict[str, object]:
    return {
        "schema_version": 2,
        "case": {
            "spatial_dim": 2,
            "coordinate_system": "cartesian_xy",
            "adapter": "comsol",
        },
        "inputs": {"comsol_manifest": "manifest.yaml"},
        "physics": {
            "gas": {
                "temperature_K": 300.0,
                "dynamic_viscosity_Pas": 1.8e-5,
                "density_kgm3": 1.0,
                "molecular_mass_amu": 39.948,
            },
            "seed": 12345,
        },
        "time": {"dt": 0.1, "t_end": 0.2},
        "output": {"mode": "standard"},
    }


def _load_test_case(root: Path):
    config_path = root / "run.yaml"
    config_path.write_text(yaml.safe_dump(_runtime_config()), encoding="utf-8")
    return load_case(config_path)


def test_manifest_v2_drives_artifacts_mapping_units_and_coordinate_scale(
    tmp_path: Path,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    manifest = ComsolCaseManifest.load(manifest_path)

    assert manifest.validate(strict=True) == []
    assert (
        manifest.provider_config()["field"]["quantity_mapping"]["ux"]["source"]
        == "raw_component_1"
    )

    runtime = _load_test_case(tmp_path)._context

    assert runtime.geometry_provider.geometry.axes[0][-1] == pytest.approx(0.01)
    assert runtime.particles.position[0].tolist() == pytest.approx([0.005, 0.005])
    assert set(runtime.field_provider.field.quantities) == {"ux", "uy"}
    assert runtime.field_provider.field.quantities["ux"].data == pytest.approx(
        np.full((2, 3, 3), 2.0, dtype=np.float64)
    )
    assert runtime.wall_catalog.part_models[0].part_id == 7


def test_manifest_rejects_field_artifact_that_runtime_cannot_load(
    tmp_path: Path,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["fields"]["velocity"]["artifact"] = "geometry"
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match="multi-artifact field loading is not supported"
    ):
        ComsolCaseManifest.load(manifest_path).validate(strict=True)


def test_manifest_rejects_enabled_force_without_semantic_field(tmp_path: Path) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["forces"].append({"solver_force": "electric", "enabled": True})
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            "enabled force 'electric' requires manifest semantic field 'electric_field'"
        ),
    ):
        ComsolCaseManifest.load(manifest_path).validate(strict=True)


def test_manifest_validation_error_order_and_strict_schema_are_stable(
    tmp_path: Path,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["coordinates"]["coordinate_scale_m_per_model_unit"] = 0.0
    payload["coordinates"]["axis_order"] = ["y", "x"]
    payload["fields"]["velocity"]["unit"] = "cm/s"
    payload["unexpected"] = True
    payload["model"]["name"] = ""
    payload["time"]["interpolation"] = "nearest"
    payload["metadata"]["vacuum_domain_ids"] = [1, 1]
    payload["metadata"]["geometry_source"] = "implicit"
    payload["metadata"]["release_boundary_projection"] = {"inward_offset_m": -1.0}
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    manifest = ComsolCaseManifest.load(manifest_path)

    errors = manifest.validate(strict=False, verify_hashes=False)

    assert errors[:9] == [
        ("coordinates.coordinate_scale_m_per_model_unit must be positive and finite"),
        "coordinates.axis_order must be ['x', 'y'], got ['y', 'x']",
        "fields.velocity.unit must be 'm/s', got 'cm/s'",
        "unknown COMSOL manifest keys: ['unexpected']",
        "model.name is required",
        "time.interpolation must be linear",
        "metadata.vacuum_domain_ids must contain unique positive integers",
        ("metadata.geometry_source must be 'explicit_comsol_vacuum_domain_selection'"),
        (
            "metadata.release_boundary_projection.inward_offset_m is obsolete; "
            "boundary releases stay on their declared entity and the solver "
            "does not treat a segment departing from it as a hit"
        ),
    ]
    assert errors[9] == (
        "metadata.release_boundary_projection.tolerance_m is required and must be "
        "numeric"
    )
    with pytest.raises(ValueError, match="Invalid COMSOL manifest") as caught:
        manifest.validate(strict=True, verify_hashes=False)
    assert str(caught.value) == "Invalid COMSOL manifest:\n" + "\n".join(
        f"- {item}" for item in errors
    )


def test_comsol_load_rejects_transient_export_outside_manifest_time_support(
    tmp_path: Path,
) -> None:
    _write_v2_bundle(tmp_path)
    config = _runtime_config()
    config["time"] = {"dt": 0.1, "t_end": 1.1}
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"field_support_s=\[0\.0, 1\.0\].*required_support_s=\[0\.0, 1\.1\]",
    ):
        load_case(config_path)


def test_manifest_v2_rejects_geometry_boundary_part_mismatch(tmp_path: Path) -> None:
    manifest_path = _write_v2_bundle(tmp_path, boundary_part_id=8)
    manifest = ComsolCaseManifest.load(manifest_path)

    with pytest.raises(ValueError, match="coverage must match exactly"):
        manifest.validate(strict=True)


def test_manifest_v2_rejects_artifact_hash_mismatch(tmp_path: Path) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    with (tmp_path / "particles.csv").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        ComsolCaseManifest.load(manifest_path).validate(strict=True)


def test_manifest_file_validation_preserves_missing_file_error(tmp_path: Path) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    missing_path = tmp_path / "particles.csv"
    missing_path.unlink()

    errors = ComsolCaseManifest.load(manifest_path).validate(strict=False)

    assert errors == [f"artifacts.release.path does not exist: {missing_path}"]


def test_manifest_file_validation_preserves_size_then_hash_error_order(
    tmp_path: Path,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    release_path = tmp_path / "particles.csv"
    original_size = release_path.stat().st_size
    original_hash = _digest(release_path)
    with release_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    errors = ComsolCaseManifest.load(manifest_path).validate(strict=False)

    assert errors == [
        "artifacts.release.size_bytes mismatch: "
        f"expected {original_size}, got {release_path.stat().st_size}",
        "artifacts.release.sha256 mismatch: "
        f"expected {original_hash}, got {_digest(release_path)}",
    ]


def test_manifest_file_schema_errors_preserve_artifact_validation_order(
    tmp_path: Path,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    release_path = tmp_path / "particles.csv"
    release_path.write_text(
        "particle_id,release_time_s,x_m,y_m,vx_mps,vy_mps,mass_kg,"
        "drag_diameter_m,charge_C,source_part_id,density_kgm3\n"
        "-1,0,0.005,0.005,0,0,1e-18,1e-6,0,7,1000\n",
        encoding="utf-8",
    )
    _rewrite_npz(
        tmp_path / "geometry.npz",
        excluded={"boundary_edge_part_ids", "boundary_triangle_part_ids"},
    )
    _rewrite_npz(
        tmp_path / "field.npz",
        excluded={"raw_component_1", "times"},
        times=np.asarray([0.0, 2.0], dtype=np.float64),
    )
    _refresh_artifact_metadata(manifest_path, "release", "geometry", "field")

    errors = ComsolCaseManifest.load(manifest_path).validate(strict=False)

    assert errors == [
        "particles.particle_id must be >= 0",
        "geometry artifact must include explicit boundary part IDs",
        "field artifact is missing manifest component arrays: ['raw_component_1']",
        "time.support_s does not match field artifact: "
        "declared=(0.0, 1.0), actual=(0.0, 2.0)",
    ]


def test_manifest_file_validation_stops_geometry_after_boundary_csv_error(
    tmp_path: Path,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    boundaries_path = tmp_path / "boundaries.csv"
    boundaries_path.write_text("part_id\n", encoding="utf-8")
    _refresh_artifact_metadata(manifest_path, "boundaries")

    errors = ComsolCaseManifest.load(manifest_path).validate(strict=False)

    assert errors == [
        f"{boundaries_path} is missing canonical boundary columns: "
        "['comsol_entity_id', 'material_id', 'material_name', 'part_name', "
        "'role', 'wall_critical_sticking_velocity_mps', "
        "'wall_diffuse_fraction', 'wall_law', 'wall_restitution', "
        "'wall_stick_probability']"
    ]


def test_manifest_file_validation_rejects_object_field_times(tmp_path: Path) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    _rewrite_npz(
        tmp_path / "field.npz",
        excluded={"times"},
        times=np.asarray([0.0, 1.0], dtype=object),
    )
    _refresh_artifact_metadata(manifest_path, "field")

    errors = ComsolCaseManifest.load(manifest_path).validate(strict=False)

    assert errors == ["Object arrays cannot be loaded when allow_pickle=False"]


def test_manifest_time_support_uses_float_resolution_not_fixed_seconds(
    tmp_path: Path,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    field_path = tmp_path / "field.npz"
    with np.load(field_path) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    arrays["times"] = np.asarray([0.0, 1.0e-18], dtype=np.float64)
    np.savez(field_path, **arrays)

    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["time"]["support_s"] = [0.0, 2.0e-18]
    payload["artifacts"]["field"]["sha256"] = _digest(field_path)
    payload["artifacts"]["field"]["size_bytes"] = field_path.stat().st_size
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"time\.support_s does not match field artifact"
    ):
        ComsolCaseManifest.load(manifest_path).validate(strict=True)


def test_manifest_v2_validates_release_rows_with_canonical_particle_contract(
    tmp_path: Path,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    release_path = tmp_path / "particles.csv"
    release_path.write_text(
        "particle_id,release_time_s,x_m,y_m,vx_mps,vy_mps,mass_kg,drag_diameter_m,charge_C,"
        "source_part_id,density_kgm3\n"
        "-1,0,0.005,0.005,0,0,1e-18,1e-6,0,7,1000\n",
        encoding="utf-8",
    )
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["artifacts"]["release"]["sha256"] = _digest(release_path)
    payload["artifacts"]["release"]["size_bytes"] = release_path.stat().st_size
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"particles\.particle_id must be >= 0"):
        ComsolCaseManifest.load(manifest_path).validate(strict=True)


def test_unified_preflight_is_pure_and_reports_runtime_boundary_coverage(
    tmp_path: Path,
) -> None:
    _write_v2_bundle(tmp_path)
    case = _load_test_case(tmp_path)
    before = sorted(path.name for path in tmp_path.iterdir())

    report = validate_case(case, detail="summary")

    assert report.passed is True
    assert report.checks["boundary_coverage"]["geometry_part_ids"] == [7]
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_unified_preflight_rejects_inconsistent_comsol_sphere_properties(
    tmp_path: Path,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    release_path = tmp_path / "particles.csv"
    particles = pd.read_csv(release_path)
    particles.loc[0, "mass_kg"] = 1.0e-18
    particles.to_csv(release_path, index=False)
    _refresh_artifact_metadata(manifest_path, "release")
    case = _load_test_case(tmp_path)

    report = validate_case(case, detail="full")

    issue = next(
        item
        for item in report.errors
        if item.code == "physics.particle.sphere_consistency"
    )
    assert issue.context["particle_ids"] == [1]
    assert issue.context["inconsistent_count"] == 1
    assert issue.context["relative_tolerance"] == pytest.approx(1.0e-3)


def test_manifest_rejects_schema_v1_without_runtime_compatibility_branch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "particles": {"release_table": "release.csv"},
                "boundaries": {"map_file": "map.csv", "wall_law_file": "walls.csv"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema_version must be 2"):
        ComsolCaseManifest.load(path)


def test_manifest_load_rejects_non_mapping_root_with_exact_path(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(["not", "a", "mapping"]), encoding="utf-8")

    with pytest.raises(
        ValueError, match="COMSOL manifest root must be a mapping"
    ) as exc_info:
        ComsolCaseManifest.load(path)

    assert str(exc_info.value) == (
        f"COMSOL manifest root must be a mapping: {path.resolve()}"
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"schema_version": 2, "fields": []},
            "COMSOL manifest fields must be a semantic-quantity mapping",
        ),
        (
            {"schema_version": 2, "fields": {}, "artifacts": []},
            "artifacts must be a mapping",
        ),
        (
            {
                "schema_version": 2,
                "fields": {},
                "artifacts": {},
                "coordinates": {},
                "forces": {"bad": True},
            },
            "COMSOL manifest forces must be a list",
        ),
    ],
)
def test_manifest_load_rejects_malformed_sections_with_stable_messages(
    tmp_path: Path,
    payload: object,
    expected: str,
) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=expected) as exc_info:
        ComsolCaseManifest.load(path)

    assert str(exc_info.value) == expected


def test_manifest_load_checks_all_field_entries_before_components(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "fields": {
                    "velocity": {"components": []},
                    "broken": [],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match=r"COMSOL manifest fields\.broken must be a mapping"
    ) as exc_info:
        ComsolCaseManifest.load(path)

    assert str(exc_info.value) == "COMSOL manifest fields.broken must be a mapping"


def test_manifest_validates_explicit_release_projection_policy(tmp_path: Path) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["metadata"]["release_boundary_projection"] = {"tolerance_m": 1.0e-10}
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    assert ComsolCaseManifest.load(manifest_path).validate(strict=True) == []

    # The displaced-release repair is gone: boundary releases stay on their
    # declared entity, so a manifest still asking for a displacement is an
    # error rather than an ignored key.
    payload["metadata"]["release_boundary_projection"]["inward_offset_m"] = 1.0e-8
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="inward_offset_m is obsolete"):
        ComsolCaseManifest.load(manifest_path).validate(strict=True)

    payload["metadata"]["release_boundary_projection"] = {"tolerance_m": 0.0}
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="tolerance_m must be positive"):
        ComsolCaseManifest.load(manifest_path).validate(strict=True)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["metadata"].pop("source_solution_number"),
            "metadata.source_solution_number must be a positive integer",
        ),
        (
            lambda payload: payload["metadata"].__setitem__("vacuum_domain_ids", []),
            "metadata.vacuum_domain_ids must be a non-empty integer list",
        ),
        (
            lambda payload: payload["metadata"].pop("geometry_source"),
            "metadata.geometry_source must be",
        ),
    ],
)
def test_manifest_requires_resolved_solution_and_vacuum_geometry_provenance(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    mutate(payload)
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        ComsolCaseManifest.load(manifest_path).validate(strict=True)


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ("model", "model has unknown keys"),
        ("coordinates", "coordinates has unknown keys"),
        ("time", "time has unknown keys"),
        ("artifact", "artifacts.release has unknown keys"),
        ("field", "fields.velocity has unknown keys"),
        ("force", r"forces\[0\].*unknown key"),
    ],
)
def test_manifest_v2_rejects_unknown_nested_keys(
    tmp_path: Path, section: str, message: str
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    target = {
        "model": payload["model"],
        "coordinates": payload["coordinates"],
        "time": payload["time"],
        "artifact": payload["artifacts"]["release"],
        "field": payload["fields"]["velocity"],
        "force": payload["forces"][0],
    }[section]
    target["obsolete_option"] = True
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        ComsolCaseManifest.load(manifest_path).validate(strict=True)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["coordinates"].__setitem__(
                "coordinate_system", "Cartesian_xy"
            ),
            "coordinates.coordinate_system must be one of",
        ),
        (
            lambda payload: payload["coordinates"].__setitem__(
                "coordinate_system", " cartesian_xy"
            ),
            "leading or trailing whitespace",
        ),
        (
            lambda payload: payload["time"].__setitem__("interpolation", "Linear"),
            "time.interpolation must be linear",
        ),
        (
            lambda payload: payload["forces"][0].__setitem__("solver_force", "Drag"),
            "solver_force.*must be one of",
        ),
        (
            lambda payload: payload["forces"][0].__setitem__("law", "STOKES"),
            "law.*must be one of",
        ),
        (
            lambda payload: payload["forces"][0].pop("enabled"),
            r"forces\[0\]\.enabled.*is required",
        ),
        (
            lambda payload: payload["artifacts"]["release"].__setitem__(
                "sha256", "A" * 64
            ),
            "lowercase SHA-256",
        ),
        (
            lambda payload: payload["artifacts"]["release"].__setitem__(
                "path", " particles.csv"
            ),
            "leading or trailing whitespace",
        ),
        (
            lambda payload: payload["model"].__setitem__("name", " unit-test"),
            "model.name must not contain leading or trailing whitespace",
        ),
    ],
)
def test_manifest_v2_rejects_noncanonical_text_instead_of_normalizing(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    mutate(payload)
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        ComsolCaseManifest.load(manifest_path).validate(strict=True)


def test_manifest_preserves_case_sensitive_human_and_component_names(
    tmp_path: Path,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["model"]["name"] = "Unit-Test Model"
    payload["fields"]["velocity"]["components"]["x"] = "Raw_Component_X"
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    manifest = ComsolCaseManifest.load(manifest_path)

    assert manifest.model["name"] == "Unit-Test Model"
    assert manifest.fields[0].semantic_quantity == "velocity"
    assert (
        manifest.provider_config()["field"]["quantity_mapping"]["ux"]["source"]
        == "Raw_Component_X"
    )


@pytest.mark.parametrize(
    "obsolete_key", ["name", "physical_quantity", "mesh", "interpolation"]
)
def test_manifest_rejects_redundant_field_contract_keys(
    tmp_path: Path,
    obsolete_key: str,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["fields"]["velocity"][obsolete_key] = {
        "name": "velocity",
        "physical_quantity": "velocity",
        "mesh": "mesh1",
        "interpolation": "linear",
    }[obsolete_key]
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(
        ValueError, match=rf"fields\.velocity has unknown keys.*{obsolete_key}"
    ):
        ComsolCaseManifest.load(manifest_path).validate(strict=True)


def test_manifest_uses_generic_scalar_semantic_key_as_provider_target(
    tmp_path: Path,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["fields"].update(
        {
            "ne": {
                "components": {"value": "raw_temperature"},
                "unit": "1/m^3",
                "scale_to_si": 1.0,
            },
            "Te": {
                "components": {"value": "raw_temperature"},
                "unit": "eV",
                "scale_to_si": 1.0,
            },
            "phi": {
                "components": {"value": "raw_temperature"},
                "unit": "V",
                "scale_to_si": 1.0,
            },
        }
    )
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    manifest = ComsolCaseManifest.load(manifest_path)
    assert manifest.validate(strict=True) == []
    mapping = manifest.provider_config()["field"]["quantity_mapping"]
    assert {"ne", "Te", "phi"}.issubset(mapping)
    assert mapping["ne"]["semantic_quantity"] == "ne"


@pytest.mark.parametrize(
    "semantic", ["scalar", "Temperature", "electron density", "9density"]
)
def test_manifest_rejects_ambiguous_generic_scalar_semantic_keys(
    tmp_path: Path,
    semantic: str,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["fields"][semantic] = {
        "components": {"value": "raw_temperature"},
        "unit": "K",
        "scale_to_si": 1.0,
    }
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be an exact built-in name"):
        ComsolCaseManifest.load(manifest_path).validate(strict=True)


def test_typed_force_inventory_reaches_catalog_plan_and_runtime_parameters(
    tmp_path: Path,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["fields"].update(
        {
            "electric_field": {
                "unit": "V/m",
                "scale_to_si": 1.0,
                "components": {
                    "x": "raw_electric_0",
                    "y": "raw_electric_1",
                },
            },
            "temperature": {
                "unit": "K",
                "scale_to_si": 1.0,
                "components": {"value": "raw_temperature"},
            },
        }
    )
    payload["forces"] = [
        {"solver_force": "drag", "enabled": True, "law": "stokes"},
        {"solver_force": "electric", "enabled": True},
        {
            "solver_force": "gravity",
            "enabled": True,
            "model": "constant_acceleration",
            "parameters": {
                "acceleration_mps2": [1.25, -9.5],
                "buoyancy": True,
            },
        },
        {
            "solver_force": "thermophoresis",
            "enabled": True,
            "model": "continuum",
            "parameters": {
                "gas_thermal_conductivity_W_mK": 0.031,
                "particle_thermal_conductivity_W_mK": 2.7,
                "Cs": 1.21,
                "Cm": 1.12,
                "Ct": 2.04,
            },
        },
        {
            "solver_force": "dielectrophoresis",
            "enabled": True,
            "model": "ac_clausius_mossotti",
            "parameters": {
                "medium_rel_permittivity": 1.7,
                "particle_rel_permittivity": 4.2,
                "medium_conductivity_Sm": 0.03,
                "particle_conductivity_Sm": 0.8,
                "frequency_Hz": 13.56e6,
            },
        },
        {
            "solver_force": "lift",
            "enabled": True,
            "model": "saffman",
            "parameters": {"coefficient": 8.75},
        },
        {
            "solver_force": "virtual_mass",
            "enabled": True,
            "model": "particle_material_acceleration",
            "parameters": {"coefficient": 0.73},
        },
    ]
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    manifest = ComsolCaseManifest.load(manifest_path)
    assert manifest.validate(strict=True) == []
    assert isinstance(manifest.force_model, ForceModel)

    case = _load_test_case(tmp_path)
    context = case._context
    by_name = context.force_catalog.by_name()
    assert context.plan.body_acceleration_mps2 == pytest.approx((1.25, -9.5))
    assert by_name["electric"].model == "particle_charge"
    assert by_name["gravity"].force.buoyancy is True
    assert by_name["thermophoresis"].model == "continuum"
    assert by_name["thermophoresis"].force.Cs == pytest.approx(1.21)
    assert by_name["dielectrophoresis"].model == "ac_clausius_mossotti"
    assert by_name["dielectrophoresis"].force.frequency_Hz == pytest.approx(13.56e6)
    assert by_name["lift"].force.coefficient == pytest.approx(8.75)
    assert by_name["virtual_mass"].force.coefficient == pytest.approx(0.73)

    runtime = context.options.force_runtime
    assert runtime.gravity_buoyancy_enabled is True
    assert runtime.thermophoresis_model == "continuum"
    assert runtime.gas_thermal_conductivity_W_mK == pytest.approx(0.031)
    assert runtime.particle_thermal_conductivity_W_mK == pytest.approx(2.7)
    assert runtime.dielectrophoresis_model == "ac_clausius_mossotti"
    assert runtime.dep_medium_rel_permittivity == pytest.approx(1.7)
    assert runtime.dep_particle_conductivity_Sm == pytest.approx(0.8)
    assert runtime.dep_frequency_Hz == pytest.approx(13.56e6)
    assert runtime.lift_coefficient == pytest.approx(8.75)
    assert runtime.virtual_mass_coefficient == pytest.approx(0.73)

    metadata = {item["name"]: item for item in case._execution["forces"]}
    assert metadata["gravity"]["parameters"]["acceleration_mps2"] == [1.25, -9.5]
    assert metadata["gravity"]["parameters"]["buoyancy"] is True
    assert metadata["thermophoresis"]["parameters"]["Cs"] == pytest.approx(1.21)
    assert metadata["dielectrophoresis"]["parameters"]["frequency_Hz"] == pytest.approx(
        13.56e6
    )
    assert metadata["virtual_mass"]["parameters"]["coefficient"] == pytest.approx(0.73)


@pytest.mark.parametrize(
    ("force", "message"),
    [
        (
            {"solver_force": "drag", "enabled": True},
            r"forces\[1\]\.law.*is required when drag is enabled",
        ),
        (
            {"solver_force": "electric", "enabled": True, "law": "particle_charge"},
            r"forces\[1\]\.law.*is valid only for drag",
        ),
        (
            {"solver_force": "electric", "enabled": True, "model": "particle_charge"},
            r"forces\[1\].*unknown key",
        ),
        (
            {"solver_force": "gravity", "enabled": True},
            "acceleration_mps2.*is required",
        ),
        (
            {
                "solver_force": "gravity",
                "enabled": True,
                "parameters": {"acceleration_mps2": [9.81]},
            },
            "must contain exactly 2 components",
        ),
        (
            {
                "solver_force": "gravity",
                "enabled": True,
                "parameters": {
                    "acceleration_mps2": [0.0, -9.81],
                    "buoyancy": "true",
                },
            },
            "buoyancy.*YAML boolean",
        ),
        (
            {
                "solver_force": "thermophoresis",
                "enabled": True,
                "parameters": {"gas_thermal_conductivity_W_mK": 0.03},
            },
            "particle_thermal_conductivity_W_mK",
        ),
        (
            {
                "solver_force": "dielectrophoresis",
                "enabled": True,
                "model": "dc",
                "parameters": {},
            },
            "medium_rel_permittivity",
        ),
        (
            {
                "solver_force": "dielectrophoresis",
                "enabled": True,
                "model": "ac_clausius_mossotti",
                "parameters": {"medium_rel_permittivity": 1.2},
            },
            "AC dielectrophoresis requires explicit",
        ),
        (
            {
                "solver_force": "lift",
                "enabled": True,
                "model": "Saffman",
            },
            "physics.forces.lift.model|forces\\[1\\]\\.model",
        ),
        (
            {
                "solver_force": "virtual_mass",
                "enabled": True,
                "parameters": {"roughness": 0.5},
            },
            "unknown key.*roughness",
        ),
    ],
)
def test_manifest_force_inventory_rejects_wrong_shape_or_missing_physics(
    tmp_path: Path,
    force: dict[str, object],
    message: str,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["forces"].append(force)
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        ComsolCaseManifest.load(manifest_path).validate(strict=True)


def test_manifest_force_inventory_rejects_duplicate_force_entries(
    tmp_path: Path,
) -> None:
    manifest_path = _write_v2_bundle(tmp_path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["forces"].append({"solver_force": "drag", "enabled": False})
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate solver_force 'drag'"):
        ComsolCaseManifest.load(manifest_path).validate(strict=True)
