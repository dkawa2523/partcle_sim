from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

import particle_tracer_unified.comsol_case.builder as builder_module
import particle_tracer_unified.comsol_case.mesh as mesh_module
from particle_tracer_unified.comsol_case import _mesh_artifacts as mesh_artifacts
from particle_tracer_unified.comsol_case import _mesh_parsing as mesh_parsing
from particle_tracer_unified.comsol_case import _mesh_topology as mesh_topology
from particle_tracer_unified.comsol_case.builder import write_case_files
from particle_tracer_unified.comsol_case.cli import main
from particle_tracer_unified.comsol_case.mesh import (
    MeshTypeBlock,
    ParsedMesh,
    assign_part_ids_from_edge_entities,
    build_precomputed_arrays,
    domain_boundary_edge_vertex_ids,
    parse_comsol_mphtxt,
    scale_mesh_coordinates,
    select_vacuum_domains,
    write_geometry_npz,
)
from particle_tracer_unified.configuration import load_run_config
from particle_tracer_unified.core.geometry2d import (
    build_boundary_loops_2d,
    validate_boundary_edges_2d,
)
from particle_tracer_unified.io.comsol_manifest import ComsolCaseManifest
from particle_tracer_unified.providers.precomputed import build_precomputed_geometry


def _write_square_mesh(path: Path) -> Path:
    path.write_text(
        """2 # sdim
4 # number of mesh vertices
# Mesh vertex coordinates
0 0
1 0
1 1
0 1
2 # number of element types
3 edg # type name
2 # number of vertices per element
4 # number of elements
# Elements
0 1
1 2
2 3
3 0
4 # number of geometric entity indices
# Geometric entity indices
0
1
2
3
4 quad # type name
4 # number of vertices per element
1 # number of elements
# Elements
0 1 2 3
1 # number of geometric entity indices
# Geometric entity indices
0
""",
        encoding="utf-8",
    )
    return path


def _write_two_domain_mesh(path: Path) -> Path:
    path.write_text(
        """2 # sdim
6 # number of mesh vertices
# Mesh vertex coordinates
0 0
1 0
2 0
0 1
1 1
2 1
2 # number of element types
3 edg # type name
2 # number of vertices per element
7 # number of elements
# Elements
0 1
1 2
2 5
5 4
4 3
3 0
1 4
7 # number of geometric entity indices
# Geometric entity indices
0
1
2
3
4
5
6
4 quad # type name
4 # number of vertices per element
2 # number of elements
# Elements
0 1 4 3
1 2 5 4
2 # number of geometric entity indices
# Geometric entity indices
0
1
""",
        encoding="utf-8",
    )
    return path


def _write_field(path: Path) -> Path:
    axis = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)
    times = np.asarray([0.0, 1.0], dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    ux = np.stack([xx + yy, xx + yy + 1.0])
    uy = np.stack([xx - yy, xx - yy + 0.5])
    np.savez_compressed(
        path,
        axis_0=axis,
        axis_1=axis,
        times=times,
        valid_mask=np.ones((3, 3), dtype=bool),
        ux=ux,
        uy=uy,
        E_x=np.ones_like(ux) * 2.0,
        E_y=np.ones_like(uy) * -3.0,
        T=np.ones_like(ux) * 300.0,
        mu=np.ones_like(ux) * 1.8e-5,
    )
    return path


def _write_boundaries(
    path: Path,
    part_ids: tuple[int, ...] = (1, 2, 3, 4),
    *,
    comsol_entity_ids: tuple[int, ...] | None = None,
) -> Path:
    entity_ids = part_ids if comsol_entity_ids is None else comsol_entity_ids
    if len(entity_ids) != len(part_ids):
        raise ValueError("part_ids and comsol_entity_ids must have equal length")
    rows = [
        {
            "part_id": part_id,
            "part_name": f"wall_{part_id}",
            "comsol_entity_id": entity_id,
            "role": "wall",
            "wall_law": "specular",
            "wall_stick_probability": 0.0,
            "wall_restitution": 1.0,
            "wall_diffuse_fraction": 0.0,
            "wall_critical_sticking_velocity_mps": 0.0,
            "material_id": 10 + part_id,
            "material_name": "steel",
            "metadata_json": "{}",
        }
        for part_id, entity_id in zip(part_ids, entity_ids, strict=True)
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_particles(path: Path, source_part_id: int = 1) -> Path:
    pd.DataFrame(
        [
            {
                "particle_id": 1,
                "release_time_s": 0.25,
                "x_m": 0.5,
                "y_m": 0.5,
                "vx_mps": 0.0,
                "vy_mps": 0.0,
                "mass_kg": 1.0e-15,
                "drag_diameter_m": 1.0e-6,
                "charge_C": -1.0e-17,
                "source_part_id": source_part_id,
                "density_kgm3": 1200.0,
            }
        ]
    ).to_csv(path, index=False)
    return path


def _two_square_mesh(*, scale: float, separated: bool) -> ParsedMesh:
    length = float(scale)
    if separated:
        gap = 0.02 * length
        vertices = np.asarray(
            [
                [0.0, 0.0],
                [length, 0.0],
                [length, length],
                [0.0, length],
                [length + gap, 0.0],
                [2.0 * length + gap, 0.0],
                [2.0 * length + gap, length],
                [length + gap, length],
            ],
            dtype=np.float64,
        )
        quads = np.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
        edges = np.asarray(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
            ],
            dtype=np.int64,
        )
    else:
        vertices = np.asarray(
            [
                [0.0, 0.0],
                [length, 0.0],
                [2.0 * length, 0.0],
                [0.0, length],
                [length, length],
                [2.0 * length, length],
            ],
            dtype=np.float64,
        )
        quads = np.asarray([[0, 1, 4, 3], [1, 2, 5, 4]], dtype=np.int64)
        edges = np.asarray(
            [[0, 1], [1, 2], [2, 5], [5, 4], [4, 3], [3, 0], [1, 4]],
            dtype=np.int64,
        )
    return ParsedMesh(
        sdim=2,
        vertices=vertices,
        type_blocks={
            "edg": MeshTypeBlock(
                type_name="edg",
                vertices_per_element=2,
                elements=edges,
                geometric_entity_indices=np.arange(edges.shape[0], dtype=np.int64),
            ),
            "quad": MeshTypeBlock(
                type_name="quad",
                vertices_per_element=4,
                elements=quads,
                geometric_entity_indices=np.zeros(quads.shape[0], dtype=np.int64),
            ),
        },
    )


def _triangle_mesh() -> ParsedMesh:
    return ParsedMesh(
        sdim=2,
        vertices=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float64,
        ),
        type_blocks={
            "edg": MeshTypeBlock(
                type_name="edg",
                vertices_per_element=2,
                elements=np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.int64),
                geometric_entity_indices=np.asarray([0, 1, 2], dtype=np.int64),
            ),
            "tri": MeshTypeBlock(
                type_name="tri",
                vertices_per_element=3,
                elements=np.asarray([[0, 1, 2]], dtype=np.int64),
                geometric_entity_indices=np.asarray([0], dtype=np.int64),
            ),
        },
    )


def _build_inputs(root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        _write_square_mesh(root / "mesh.mphtxt"),
        _write_field(root / "field.npz"),
        _write_particles(root / "release.csv"),
        _write_boundaries(root / "wall_contract.csv"),
    )


def _build_case(root: Path, out: Path, **updates: object) -> None:
    mesh, field, release, boundaries = _build_inputs(root)
    arguments: dict[str, object] = {
        "field_bundle_path": field,
        "release_table_path": release,
        "boundaries_path": boundaries,
        "diagnostic_grid_spacing_m": 0.5,
        "coordinate_scale_m_per_model_unit": 1.0,
        "coordinate_system": "cartesian_xy",
        "model_name": "square-test",
        "study": "std1",
        "dataset": "dset1",
        "solution": "sol1",
        "solution_number": 1,
        "vacuum_domain_ids": (1,),
        "drag_law": "stokes",
        "enabled_forces": ("electric",),
        "gas_dynamic_viscosity_Pas": 1.8e-5,
        "solver_dt_s": 0.1,
        "solver_t_end_s": 1.0,
    }
    arguments.update(updates)
    write_case_files(mesh, out, **arguments)


def test_geometry_only_writes_no_runtime_contract(tmp_path: Path) -> None:
    mesh = _write_square_mesh(tmp_path / "mesh.mphtxt")
    out = tmp_path / "case"

    write_case_files(
        mesh,
        out,
        geometry_only=True,
        diagnostic_grid_spacing_m=0.5,
        coordinate_scale_m_per_model_unit=1.0,
        vacuum_domain_ids=(1,),
    )

    assert (out / "generated" / "comsol_geometry_2d.npz").is_file()
    assert not (out / "run_config.yaml").exists()
    assert not (out / "comsol_manifest.yaml").exists()
    with np.load(out / "generated" / "comsol_geometry_2d.npz") as geometry:
        metadata = json.loads(str(np.asarray(geometry["metadata_json"]).item()))
    topology = metadata["boundary_edge_topology"]
    assert topology["identity_policy"] == "geometry-scaled-float64-v1"
    assert topology["identity_resolution_m"] == pytest.approx(1.0)


def test_geometry_only_rejects_field_bundle_before_output(tmp_path: Path) -> None:
    mesh, field, _, _ = _build_inputs(tmp_path)
    out = tmp_path / "case"

    with pytest.raises(
        ValueError,
        match="geometry_only cannot be combined with a field bundle",
    ):
        write_case_files(
            mesh,
            out,
            field_bundle_path=field,
            geometry_only=True,
            diagnostic_grid_spacing_m=0.5,
            coordinate_scale_m_per_model_unit=1.0,
            vacuum_domain_ids=(1,),
        )

    assert not out.exists()


def test_runnable_missing_inputs_keep_public_error_order(tmp_path: Path) -> None:
    mesh, field, _, _ = _build_inputs(tmp_path)
    out = tmp_path / "case"

    with pytest.raises(
        ValueError,
        match="runnable COMSOL case requires explicit inputs",
    ) as exc_info:
        write_case_files(
            mesh,
            out,
            field_bundle_path=field,
            diagnostic_grid_spacing_m=0.5,
            coordinate_scale_m_per_model_unit=1.0,
            vacuum_domain_ids=(1,),
        )

    assert str(exc_info.value) == (
        "runnable COMSOL case requires explicit inputs: "
        "['release_table_path', 'boundaries_path', 'model_name', 'study', "
        "'dataset', 'solution', 'solution_number', 'drag_law']"
    )
    assert not out.exists()


def test_cli_rejects_geometry_only_with_field_bundle_before_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mesh, field, _, _ = _build_inputs(tmp_path)
    out = tmp_path / "case"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--mphtxt",
                str(mesh),
                "--field-bundle",
                str(field),
                "--out-dir",
                str(out),
                "--geometry-only",
                "--diagnostic-grid-spacing-m",
                "0.5",
                "--coordinate-scale-m-per-model-unit",
                "1.0",
                "--vacuum-domain-id",
                "1",
            ]
        )

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "--geometry-only cannot be combined with --field-bundle" in stderr
    assert not out.exists()


def test_cli_rejects_geometry_only_with_raw_export_before_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "case"

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--raw-export-dir",
                str(tmp_path / "missing-export"),
                "--out-dir",
                str(out),
                "--profile",
                "icp_cf4_o2",
                "--geometry-only",
                "--diagnostic-grid-spacing-m",
                "0.5",
            ]
        )

    assert exc_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "--geometry-only cannot be combined with --raw-export-dir" in stderr
    assert not out.exists()


def test_runnable_artifact_stages_keep_their_dependency_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def record_call(name: str) -> None:
        original = getattr(builder_module, name)

        def wrapped(*args: object, **kwargs: object):
            events.append(name)
            return original(*args, **kwargs)

        monkeypatch.setattr(builder_module, name, wrapped)

    for stage in (
        "pack_field_bundle",
        "write_geometry_npz",
        "write_comsol_entity_maps",
        "build_summary",
        "canonical_boundary_table",
        "canonical_release_table",
        "copy_explicit_input",
        "validate_gas",
        "write_case_contract",
    ):
        record_call(stage)

    _build_case(tmp_path, tmp_path / "case")

    assert events == [
        "canonical_boundary_table",
        "pack_field_bundle",
        "write_geometry_npz",
        "write_comsol_entity_maps",
        "build_summary",
        "canonical_release_table",
        "copy_explicit_input",
        "copy_explicit_input",
        "validate_gas",
        "write_case_contract",
    ]


def test_runnable_validation_keeps_solution_and_projection_contracts(
    tmp_path: Path,
) -> None:
    invalid_out = tmp_path / "invalid"
    with pytest.raises(
        ValueError,
        match="solution_number must be a positive integer",
    ):
        _build_case(tmp_path, invalid_out, solution_number=0)
    assert not invalid_out.exists()

    # Declaring the detection tolerance is what tells the adapter boundary
    # releases are expected; they are then snapped onto their declared entity
    # and left there, which is where COMSOL puts an inlet particle.
    valid_out = tmp_path / "valid"
    _build_case(
        tmp_path,
        valid_out,
        release_projection_tolerance_m=2.0e-6,
    )
    manifest = yaml.safe_load(
        (valid_out / "comsol_manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest["metadata"]["release_boundary_projection"] == {
        "tolerance_m": 2.0e-6,
    }


@pytest.mark.parametrize(
    ("missing_name", "message"),
    [
        ("diagnostic_grid_spacing_m", "diagnostic_grid_spacing_m is required"),
        (
            "coordinate_scale_m_per_model_unit",
            "coordinate_scale_m_per_model_unit is required",
        ),
        ("solver_dt_s", "solver_dt_s.*required"),
        ("solver_t_end_s", "solver_t_end_s.*required"),
    ],
)
def test_builder_rejects_missing_physical_and_time_scales_before_output(
    tmp_path: Path,
    missing_name: str,
    message: str,
) -> None:
    mesh, field, release, boundaries = _build_inputs(tmp_path)
    out = tmp_path / "case"
    arguments: dict[str, object] = {
        "field_bundle_path": field,
        "release_table_path": release,
        "boundaries_path": boundaries,
        "diagnostic_grid_spacing_m": 0.5,
        "coordinate_scale_m_per_model_unit": 1.0,
        "model_name": "square-test",
        "study": "std1",
        "dataset": "dset1",
        "solution": "sol1",
        "solution_number": 1,
        "vacuum_domain_ids": (1,),
        "drag_law": "none",
        "solver_dt_s": 0.1,
        "solver_t_end_s": 1.0,
    }
    arguments[missing_name] = None

    with pytest.raises(ValueError, match=message):
        write_case_files(mesh, out, **arguments)

    assert not out.exists()


@pytest.mark.parametrize(
    ("invalid_name", "invalid_value"),
    [
        ("diagnostic_grid_spacing_m", 0.0),
        ("coordinate_scale_m_per_model_unit", float("nan")),
        ("solver_dt_s", float("inf")),
        ("solver_t_end_s", -1.0),
    ],
)
def test_builder_rejects_invalid_physical_and_time_scales_before_output(
    tmp_path: Path,
    invalid_name: str,
    invalid_value: float,
) -> None:
    mesh, field, release, boundaries = _build_inputs(tmp_path)
    out = tmp_path / "case"
    arguments: dict[str, object] = {
        "field_bundle_path": field,
        "release_table_path": release,
        "boundaries_path": boundaries,
        "diagnostic_grid_spacing_m": 0.5,
        "coordinate_scale_m_per_model_unit": 1.0,
        "model_name": "square-test",
        "study": "std1",
        "dataset": "dset1",
        "solution": "sol1",
        "solution_number": 1,
        "vacuum_domain_ids": (1,),
        "drag_law": "none",
        "solver_dt_s": 0.1,
        "solver_t_end_s": 1.0,
    }
    arguments[invalid_name] = invalid_value

    with pytest.raises(ValueError, match=rf"{invalid_name}.*positive and finite"):
        write_case_files(mesh, out, **arguments)

    assert not out.exists()


@pytest.mark.parametrize(
    "omitted_option",
    ["--diagnostic-grid-spacing-m", "--coordinate-scale-m-per-model-unit"],
)
def test_builder_cli_requires_geometry_scales_before_output(
    tmp_path: Path,
    omitted_option: str,
) -> None:
    mesh = _write_square_mesh(tmp_path / "mesh.mphtxt")
    out = tmp_path / "case"
    options = {
        "--diagnostic-grid-spacing-m": "0.5",
        "--coordinate-scale-m-per-model-unit": "1.0",
    }
    argv = [
        "--mphtxt",
        str(mesh),
        "--out-dir",
        str(out),
        "--geometry-only",
        "--vacuum-domain-id",
        "1",
    ]
    for option, value in options.items():
        if option != omitted_option:
            argv.extend((option, value))

    with pytest.raises(SystemExit):
        main(argv)

    assert not out.exists()


@pytest.mark.parametrize("omitted_option", ["--dt-s", "--t-end-s"])
def test_builder_cli_requires_runnable_time_scales_before_output(
    tmp_path: Path,
    omitted_option: str,
) -> None:
    out = tmp_path / "case"
    options = {"--dt-s": "0.1", "--t-end-s": "1.0"}
    argv = [
        "--out-dir",
        str(out),
        "--diagnostic-grid-spacing-m",
        "0.5",
        "--coordinate-scale-m-per-model-unit",
        "1.0",
    ]
    for option, value in options.items():
        if option != omitted_option:
            argv.extend((option, value))

    with pytest.raises(SystemExit):
        main(argv)

    assert not out.exists()


def test_builder_emits_only_v2_case_inputs_and_manifest_is_self_consistent(
    tmp_path: Path,
) -> None:
    out = tmp_path / "case"
    _build_case(tmp_path, out)

    assert (out / "particles.csv").is_file()
    assert (out / "boundaries.csv").is_file()
    assert not (out / "materials.csv").exists()
    assert not (out / "part_walls.csv").exists()

    config_payload = yaml.safe_load(
        (out / "run_config.yaml").read_text(encoding="utf-8")
    )
    assert set(config_payload) == {
        "schema_version",
        "case",
        "inputs",
        "physics",
        "time",
        "output",
    }
    assert config_payload["case"]["adapter"] == "comsol"
    assert config_payload["inputs"] == {"comsol_manifest": "comsol_manifest.yaml"}
    assert "drag" not in config_payload["physics"]
    assert config_payload["physics"]["forces"] == {}
    assert load_run_config(out / "run_config.yaml").case.adapter == "comsol"

    manifest_payload = yaml.safe_load(
        (out / "comsol_manifest.yaml").read_text(encoding="utf-8")
    )
    assert manifest_payload["schema_version"] == 2
    assert manifest_payload["model"] == {
        "name": "square-test",
        "study": "std1",
        "dataset": "dset1",
        "solution": "sol1",
    }
    assert manifest_payload["time"] == {
        "interpolation": "linear",
        "support_s": [0.0, 1.0],
    }
    assert manifest_payload["coordinates"]["axis_order"] == ["x", "y"]
    assert manifest_payload["fields"]["velocity"]["components"] == {
        "x": "ux",
        "y": "uy",
    }
    assert manifest_payload["fields"]["electric_field"]["components"] == {
        "x": "E_x",
        "y": "E_y",
    }
    for semantic, field_spec in manifest_payload["fields"].items():
        assert semantic
        assert not (
            {"name", "physical_quantity", "mesh", "interpolation"} & set(field_spec)
        )
    assert manifest_payload["forces"] == [
        {"solver_force": "drag", "enabled": True, "law": "stokes"},
        {"solver_force": "electric", "enabled": True},
    ]
    assert manifest_payload["metadata"]["vacuum_domain_ids"] == [1]
    assert manifest_payload["metadata"]["source_solution_number"] == 1
    assert (
        manifest_payload["metadata"]["geometry_source"]
        == "explicit_comsol_vacuum_domain_selection"
    )
    for artifact in manifest_payload["artifacts"].values():
        path = out / artifact["path"]
        assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert artifact["size_bytes"] == path.stat().st_size
    assert (
        ComsolCaseManifest.load(out / "comsol_manifest.yaml").validate(strict=True)
        == []
    )


def test_builder_writes_debug_output_interval_to_typed_config(tmp_path: Path) -> None:
    out = tmp_path / "debug-case"

    _build_case(
        tmp_path,
        out,
        output_mode="debug",
        trajectory_interval_steps=3,
    )

    config = yaml.safe_load((out / "run_config.yaml").read_text(encoding="utf-8"))
    assert config["output"] == {
        "mode": "debug",
        "trajectory_interval_steps": 3,
    }


def test_builder_rejects_incomplete_boundary_coverage(tmp_path: Path) -> None:
    mesh, field, release, _ = _build_inputs(tmp_path)
    boundaries = _write_boundaries(tmp_path / "incomplete.csv", (1, 2, 3))

    with pytest.raises(
        ValueError, match="explicitly cover every generated geometry part"
    ):
        write_case_files(
            mesh,
            tmp_path / "case",
            field_bundle_path=field,
            release_table_path=release,
            boundaries_path=boundaries,
            diagnostic_grid_spacing_m=0.5,
            coordinate_scale_m_per_model_unit=1.0,
            model_name="square-test",
            study="std1",
            dataset="dset1",
            solution="sol1",
            solution_number=1,
            vacuum_domain_ids=(1,),
            drag_law="none",
            solver_dt_s=0.1,
            solver_t_end_s=1.0,
        )


def test_builder_rejects_missing_drag_gas_instead_of_filling_defaults(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires explicit gas values"):
        _build_case(tmp_path, tmp_path / "case", gas_dynamic_viscosity_Pas=None)


def test_builder_rejects_release_part_without_boundary_registration(
    tmp_path: Path,
) -> None:
    mesh, field, _, boundaries = _build_inputs(tmp_path)
    release = _write_particles(tmp_path / "unknown_source.csv", source_part_id=99)

    with pytest.raises(ValueError, match="unregistered source_part_id"):
        write_case_files(
            mesh,
            tmp_path / "case",
            field_bundle_path=field,
            release_table_path=release,
            boundaries_path=boundaries,
            diagnostic_grid_spacing_m=0.5,
            coordinate_scale_m_per_model_unit=1.0,
            model_name="square-test",
            study="std1",
            dataset="dset1",
            solution="sol1",
            solution_number=1,
            vacuum_domain_ids=(1,),
            drag_law="none",
            solver_dt_s=0.1,
            solver_t_end_s=1.0,
        )


def test_builder_maps_swapped_comsol_entities_to_solver_part_ids_once(
    tmp_path: Path,
) -> None:
    mesh, field, _, _ = _build_inputs(tmp_path)
    release = _write_particles(tmp_path / "mapped_release.csv", source_part_id=40)
    boundaries = _write_boundaries(
        tmp_path / "mapped_boundaries.csv",
        (40, 10, 30, 20),
        comsol_entity_ids=(1, 2, 3, 4),
    )
    out = tmp_path / "mapped_case"

    write_case_files(
        mesh,
        out,
        field_bundle_path=field,
        release_table_path=release,
        boundaries_path=boundaries,
        diagnostic_grid_spacing_m=0.5,
        coordinate_scale_m_per_model_unit=1.0,
        model_name="square-test",
        study="std1",
        dataset="dset1",
        solution="sol1",
        solution_number=1,
        vacuum_domain_ids=(1,),
        drag_law="none",
        solver_dt_s=0.1,
        solver_t_end_s=1.0,
    )

    with np.load(out / "generated" / "comsol_geometry_2d.npz") as geometry:
        np.testing.assert_array_equal(
            geometry["boundary_edge_part_ids"],
            np.asarray([40, 10, 30, 20], dtype=np.int32),
        )
        assert set(np.unique(geometry["nearest_boundary_part_id_map"])) <= {
            10,
            20,
            30,
            40,
        }
    mapping = pd.read_csv(out / "generated" / "comsol_boundary_entity_mapping.csv")
    assert mapping[["comsol_edge_entity_id", "solver_part_id"]].to_dict("records") == [
        {"comsol_edge_entity_id": 1, "solver_part_id": 40},
        {"comsol_edge_entity_id": 2, "solver_part_id": 10},
        {"comsol_edge_entity_id": 3, "solver_part_id": 30},
        {"comsol_edge_entity_id": 4, "solver_part_id": 20},
    ]
    assert list(pd.read_csv(out / "boundaries.csv")) == list(pd.read_csv(boundaries))


def test_builder_reports_missing_then_extra_comsol_boundary_entities(
    tmp_path: Path,
) -> None:
    mesh, field, release, _ = _build_inputs(tmp_path)
    boundaries = _write_boundaries(
        tmp_path / "wrong_entities.csv",
        (10, 20, 30, 50),
        comsol_entity_ids=(1, 2, 3, 5),
    )

    with pytest.raises(
        ValueError,
        match="explicitly cover every generated geometry part",
    ) as exc_info:
        write_case_files(
            mesh,
            tmp_path / "wrong_case",
            field_bundle_path=field,
            release_table_path=release,
            boundaries_path=boundaries,
            diagnostic_grid_spacing_m=0.5,
            coordinate_scale_m_per_model_unit=1.0,
            model_name="square-test",
            study="std1",
            dataset="dset1",
            solution="sol1",
            solution_number=1,
            vacuum_domain_ids=(1,),
            drag_law="none",
            solver_dt_s=0.1,
            solver_t_end_s=1.0,
        )

    assert str(exc_info.value).endswith("missing=[4], stale=[5]")


def test_builder_simple_force_flag_rejects_coefficient_bearing_force_before_writing(
    tmp_path: Path,
) -> None:
    out = tmp_path / "case"

    with pytest.raises(
        ValueError, match=r"--force only supports electric.*--force-inventory"
    ):
        _build_case(tmp_path, out, enabled_forces=("gravity",))

    assert not out.exists()


def test_builder_copies_validated_typed_force_inventory_without_losing_parameters(
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "forces.yaml"
    inventory.write_text(
        yaml.safe_dump(
            {
                "forces": [
                    {
                        "solver_force": "gravity",
                        "enabled": True,
                        "parameters": {
                            "acceleration_mps2": [0.0, -9.81],
                            "buoyancy": True,
                        },
                    },
                    {
                        "solver_force": "thermophoresis",
                        "enabled": True,
                        "model": "continuum",
                        "parameters": {
                            "gas_thermal_conductivity_W_mK": 0.031,
                            "particle_thermal_conductivity_W_mK": 2.4,
                        },
                    },
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "case"

    _build_case(tmp_path, out, force_inventory_path=inventory)

    manifest = ComsolCaseManifest.load(out / "comsol_manifest.yaml")
    assert manifest.validate(strict=True) == []
    model = manifest.force_model
    assert model.drag.model == "stokes"
    assert model.electric.enabled
    assert model.gravity.acceleration_mps2 == (0.0, -9.81)
    assert model.gravity.buoyancy
    assert model.thermophoresis.model == "continuum"
    assert model.thermophoresis.gas_thermal_conductivity_W_mK == pytest.approx(0.031)
    assert model.thermophoresis.particle_thermal_conductivity_W_mK == pytest.approx(2.4)


def test_builder_typed_inventory_fails_on_missing_force_physics_before_writing(
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "forces.yaml"
    inventory.write_text(
        yaml.safe_dump(
            {
                "forces": [
                    {
                        "solver_force": "thermophoresis",
                        "enabled": True,
                        "parameters": {
                            "gas_thermal_conductivity_W_mK": 0.031,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "case"

    with pytest.raises(ValueError, match="particle_thermal_conductivity_W_mK"):
        _build_case(tmp_path, out, force_inventory_path=inventory)

    assert not out.exists()


def test_builder_rejects_rz_lift_from_typed_inventory_before_writing(
    tmp_path: Path,
) -> None:
    inventory = tmp_path / "forces.yaml"
    inventory.write_text(
        yaml.safe_dump(
            {
                "forces": [
                    {
                        "solver_force": "lift",
                        "enabled": True,
                        "model": "saffman",
                        "parameters": {"coefficient": 6.46},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "case"

    with pytest.raises(ValueError, match=r"axisymmetric_rz no-swirl.*lift"):
        _build_case(
            tmp_path,
            out,
            coordinate_system="axisymmetric_rz",
            force_inventory_path=inventory,
        )

    assert not out.exists()


def test_builder_requires_explicit_vacuum_domain_before_writing(tmp_path: Path) -> None:
    mesh = _write_square_mesh(tmp_path / "mesh.mphtxt")
    out = tmp_path / "case"

    with pytest.raises(ValueError, match="explicit vacuum_domain_id"):
        write_case_files(
            mesh,
            out,
            geometry_only=True,
            diagnostic_grid_spacing_m=0.5,
            coordinate_scale_m_per_model_unit=1.0,
        )

    assert not out.exists()


def test_field_valid_mask_never_replaces_physical_geometry(tmp_path: Path) -> None:
    mesh, field, release, boundaries = _build_inputs(tmp_path)
    with np.load(field) as source:
        payload = {name: np.asarray(source[name]) for name in source.files}
    payload["valid_mask"] = np.ones((3, 3), dtype=bool)
    payload["valid_mask"][1, 1] = False
    np.savez_compressed(field, **payload)
    out = tmp_path / "case"

    write_case_files(
        mesh,
        out,
        field_bundle_path=field,
        release_table_path=release,
        boundaries_path=boundaries,
        diagnostic_grid_spacing_m=0.5,
        coordinate_scale_m_per_model_unit=1.0,
        model_name="square-test",
        study="std1",
        dataset="dset1",
        solution="sol1",
        solution_number=1,
        vacuum_domain_ids=(1,),
        drag_law="none",
        solver_dt_s=0.1,
        solver_t_end_s=1.0,
    )

    with np.load(out / "generated" / "comsol_geometry_2d.npz") as geometry:
        assert geometry["boundary_edges"].shape == (4, 2, 2)
        metadata = json.loads(str(np.asarray(geometry["metadata_json"]).item()))
        assert metadata["source_kind"] == "comsol_selected_vacuum_domain_geometry"
        assert metadata["field_support_is_physical_boundary"] is False
    with np.load(out / "generated" / "comsol_field_2d.npz") as packed_field:
        assert np.count_nonzero(packed_field["valid_mask"]) == 8


def test_vacuum_domain_selection_preserves_internal_solid_interface(
    tmp_path: Path,
) -> None:
    mesh = _write_two_domain_mesh(tmp_path / "mesh.mphtxt")
    out = tmp_path / "case"

    write_case_files(
        mesh,
        out,
        geometry_only=True,
        diagnostic_grid_spacing_m=0.5,
        coordinate_scale_m_per_model_unit=1.0,
        vacuum_domain_ids=(1,),
    )

    with np.load(out / "generated" / "comsol_geometry_2d.npz") as geometry:
        edges = np.asarray(geometry["boundary_edges"], dtype=np.float64)
        assert edges.shape == (4, 2, 2)
        assert float(np.max(edges[:, :, 0])) == pytest.approx(1.0)
        assert any(np.allclose(edge[:, 0], 1.0) for edge in edges)
    domains = pd.read_csv(out / "generated" / "comsol_domain_entity_mapping.csv")
    assert domains.set_index("comsol_domain_entity_id")[
        "selected_as_vacuum_domain"
    ].to_dict() == {
        1: True,
        2: False,
    }


def test_adjacent_vacuum_domains_keep_shared_entity_out_of_containment(
    tmp_path: Path,
) -> None:
    mesh = _write_two_domain_mesh(tmp_path / "mesh.mphtxt")
    out = tmp_path / "case"

    write_case_files(
        mesh,
        out,
        geometry_only=True,
        diagnostic_grid_spacing_m=0.5,
        coordinate_scale_m_per_model_unit=1.0,
        vacuum_domain_ids=(1, 2),
    )

    with np.load(out / "generated" / "comsol_geometry_2d.npz") as geometry:
        edges = np.asarray(geometry["boundary_edges"], dtype=np.float64)
        part_ids = np.asarray(geometry["boundary_edge_part_ids"], dtype=np.int32)
        loops = np.asarray(geometry["boundary_loops_2d_flat"], dtype=np.float64)
        nearest = np.asarray(geometry["nearest_boundary_part_id_map"], dtype=np.int32)
        inside = np.asarray(geometry["valid_mask"], dtype=bool)
        metadata = json.loads(str(np.asarray(geometry["metadata_json"]).item()))

    assert edges.shape == (7, 2, 2)
    assert 7 in part_ids
    assert loops.shape == (6, 2)
    assert np.all(inside)
    assert 7 not in nearest
    assert metadata["containment_boundary_edge_count"] == 6
    assert metadata["internal_interface_edge_count"] == 1

    loaded = build_precomputed_geometry(
        {"npz_path": str(out / "generated" / "comsol_geometry_2d.npz")},
        spatial_dim=2,
        coordinate_system="cartesian_xy",
    ).geometry
    assert loaded.boundary_edges is not None
    assert loaded.boundary_edges.shape == (7, 2, 2)
    assert len(loaded.boundary_loops_2d) == 1
    assert loaded.metadata["boundary_edge_topology"]["edge_count"] == 6


def test_vacuum_domain_selection_rejects_unknown_domain_before_writing(
    tmp_path: Path,
) -> None:
    mesh = _write_two_domain_mesh(tmp_path / "mesh.mphtxt")
    out = tmp_path / "case"

    with pytest.raises(ValueError, match=r"missing=\[3\].*available=\[1, 2\]"):
        write_case_files(
            mesh,
            out,
            geometry_only=True,
            diagnostic_grid_spacing_m=0.5,
            coordinate_scale_m_per_model_unit=1.0,
            vacuum_domain_ids=(3,),
        )

    assert not out.exists()


@pytest.mark.parametrize(
    ("domain_ids", "message"),
    [
        ((), "at least one explicit vacuum_domain_id is required"),
        ((True,), "vacuum_domain_ids must contain integers"),
        ((0,), "vacuum_domain_ids must contain positive integers"),
        ((1, 1), "vacuum_domain_ids must not contain duplicates"),
    ],
)
def test_vacuum_domain_selection_validation_contract(
    domain_ids: tuple[int | bool, ...],
    message: str,
) -> None:
    mesh = _two_square_mesh(scale=1.0, separated=True)

    with pytest.raises(ValueError, match=message):
        select_vacuum_domains(mesh, domain_ids)


def test_vacuum_selection_keeps_mesh_order_and_non_surface_blocks() -> None:
    source = _two_square_mesh(scale=1.0, separated=True)
    quad = source.type_blocks["quad"]
    mesh = ParsedMesh(
        sdim=source.sdim,
        vertices=source.vertices,
        type_blocks={
            **source.type_blocks,
            "quad": MeshTypeBlock(
                type_name=quad.type_name,
                vertices_per_element=quad.vertices_per_element,
                elements=quad.elements,
                geometric_entity_indices=np.asarray([1, 0], dtype=np.int64),
            ),
        },
    )

    selected, domain_ids = select_vacuum_domains(mesh, (2, 1))

    assert domain_ids == (1, 2)
    assert selected.type_blocks["edg"] is mesh.type_blocks["edg"]
    np.testing.assert_array_equal(
        selected.type_blocks["quad"].elements,
        mesh.type_blocks["quad"].elements,
    )
    np.testing.assert_array_equal(
        selected.type_blocks["quad"].geometric_entity_indices,
        np.asarray([1, 0], dtype=np.int64),
    )


def test_mphtxt_parser_preserves_block_order_shapes_and_numeric_dtypes(
    tmp_path: Path,
) -> None:
    parsed = parse_comsol_mphtxt(_write_square_mesh(tmp_path / "mesh.mphtxt"))

    assert parsed.sdim == 2
    assert list(parsed.type_blocks) == ["edg", "quad"]
    assert parsed.vertices.shape == (4, 2)
    assert parsed.vertices.dtype == np.float64
    for name, shape in (("edg", (4, 2)), ("quad", (1, 4))):
        block = parsed.type_blocks[name]
        assert block.type_name == name
        assert block.elements.shape == shape
        assert block.elements.dtype == np.int64
        assert block.geometric_entity_indices.dtype == np.int64

    unchanged = scale_mesh_coordinates(parsed, 1.0)
    scaled = scale_mesh_coordinates(parsed, 0.25)
    assert unchanged is parsed
    assert scaled.type_blocks is parsed.type_blocks
    assert scaled.vertices.dtype == np.float64
    np.testing.assert_array_equal(scaled.vertices, parsed.vertices * 0.25)


def test_mesh_facade_directly_reexports_each_public_owner() -> None:
    assert MeshTypeBlock is mesh_parsing.MeshTypeBlock
    assert ParsedMesh is mesh_parsing.ParsedMesh
    assert parse_comsol_mphtxt is mesh_parsing.parse_comsol_mphtxt
    assert scale_mesh_coordinates is mesh_parsing.scale_mesh_coordinates
    assert select_vacuum_domains is mesh_parsing.select_vacuum_domains
    assert (
        assign_part_ids_from_edge_entities
        is mesh_topology.assign_part_ids_from_edge_entities
    )
    assert build_precomputed_arrays is mesh_topology.build_precomputed_arrays
    assert (
        domain_boundary_edge_vertex_ids is mesh_topology.domain_boundary_edge_vertex_ids
    )
    assert (
        mesh_module.write_comsol_entity_maps is mesh_artifacts.write_comsol_entity_maps
    )
    assert mesh_module.write_geometry_npz is mesh_artifacts.write_geometry_npz


def test_mphtxt_parser_keeps_marker_and_entity_count_errors(
    tmp_path: Path,
) -> None:
    missing_marker = tmp_path / "missing.mphtxt"
    missing_marker.write_text("2 # sdim\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not find marker: # number of mesh"):
        parse_comsol_mphtxt(missing_marker)

    mismatch = _write_square_mesh(tmp_path / "mismatch.mphtxt")
    text = mismatch.read_text(encoding="utf-8").replace(
        "4 # number of geometric entity indices",
        "3 # number of geometric entity indices",
        1,
    )
    mismatch.write_text(text, encoding="utf-8")
    with pytest.raises(
        ValueError,
        match="Geometric entity size mismatch for edg: 3 vs 4",
    ):
        parse_comsol_mphtxt(mismatch)


@pytest.mark.parametrize("scale", [1.0e-13, 1.0, 1.0e3])
def test_comsol_edge_topology_and_part_assignment_are_scale_invariant(
    scale: float,
) -> None:
    mesh = _two_square_mesh(scale=scale, separated=False)

    boundary_vertex_ids = domain_boundary_edge_vertex_ids(
        mesh.vertices,
        mesh.type_blocks,
    )
    part_ids = assign_part_ids_from_edge_entities(mesh.type_blocks, boundary_vertex_ids)
    edge_keys = {tuple(sorted((int(a), int(b)))) for a, b in boundary_vertex_ids}

    assert edge_keys == {(0, 1), (1, 2), (2, 5), (4, 5), (3, 4), (0, 3)}
    assert set(part_ids.tolist()) == {1, 2, 3, 4, 5, 6}
    assert 7 not in part_ids  # The truly shared edge (1, 4) is not a domain boundary.

    arrays = build_precomputed_arrays(mesh, diagnostic_grid_spacing_m=0.5 * scale)
    normalized = np.asarray(arrays["boundary_edges"], dtype=np.float64) / scale
    assert normalized.shape == (7, 2, 2)
    assert int(arrays["containment_boundary_edge_count"]) == 6
    assert int(arrays["internal_interface_edge_count"]) == 1
    assert 7 in np.asarray(arrays["boundary_part_ids"], dtype=np.int32)
    assert len(arrays["boundary_loops_2d"]) == 1
    np.testing.assert_allclose(
        np.diff(np.asarray(arrays["axes_x"], dtype=np.float64)) / scale,
        0.5,
        rtol=0.0,
        atol=64.0 * np.finfo(np.float64).eps,
    )
    np.testing.assert_allclose(
        np.diff(np.asarray(arrays["axes_y"], dtype=np.float64)) / scale,
        0.5,
        rtol=0.0,
        atol=64.0 * np.finfo(np.float64).eps,
    )
    assert float(np.asarray(arrays["sdf"], dtype=np.float64)[1, 1]) < 0.0


def test_comsol_boundary_edges_and_part_ids_keep_surface_traversal_order() -> None:
    mesh = _two_square_mesh(scale=1.0, separated=False)

    boundary_vertex_ids = domain_boundary_edge_vertex_ids(
        mesh.vertices,
        mesh.type_blocks,
    )
    part_ids = assign_part_ids_from_edge_entities(mesh.type_blocks, boundary_vertex_ids)

    np.testing.assert_array_equal(
        boundary_vertex_ids,
        np.asarray(
            [[0, 1], [4, 3], [3, 0], [1, 2], [2, 5], [5, 4]],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        part_ids,
        np.asarray([1, 5, 6, 2, 3, 4], dtype=np.int32),
    )


@given(
    scale=st.floats(
        min_value=1.0e-12,
        max_value=1.0e6,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )
)
@settings(max_examples=24, deadline=None)
def test_comsol_edge_ownership_order_is_independent_of_coordinate_scale(
    scale: float,
) -> None:
    mesh = _two_square_mesh(scale=scale, separated=False)

    boundary_vertex_ids = domain_boundary_edge_vertex_ids(
        mesh.vertices,
        mesh.type_blocks,
    )
    part_ids = assign_part_ids_from_edge_entities(
        mesh.type_blocks,
        boundary_vertex_ids,
    )

    np.testing.assert_array_equal(
        boundary_vertex_ids,
        np.asarray(
            [[0, 1], [4, 3], [3, 0], [1, 2], [2, 5], [5, 4]],
            dtype=np.int64,
        ),
    )
    np.testing.assert_array_equal(
        part_ids,
        np.asarray([1, 5, 6, 2, 3, 4], dtype=np.int32),
    )


def test_triangle_only_topology_keeps_order_dtypes_and_entity_adjacency() -> None:
    mesh = _triangle_mesh()

    arrays = build_precomputed_arrays(mesh, diagnostic_grid_spacing_m=0.5)

    np.testing.assert_array_equal(
        arrays["boundary_edges"],
        mesh.vertices[np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.int64)],
    )
    np.testing.assert_array_equal(
        arrays["boundary_part_ids"],
        np.asarray([1, 2, 3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        arrays["triangles"],
        np.asarray([[0, 1, 2]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        arrays["triangle_part_ids"],
        np.asarray([1], dtype=np.int32),
    )
    assert np.asarray(arrays["quads"]).shape == (0, 4)
    assert np.asarray(arrays["quad_part_ids"]).dtype == np.int32

    boundary_rows = mesh_artifacts._comsol_boundary_entity_rows(mesh)
    assert [row["adjacent_domain_ids"] for row in boundary_rows] == ["1", "1", "1"]
    domain_rows = mesh_artifacts._comsol_domain_entity_rows(
        mesh,
        vacuum_domain_ids=(1,),
    )
    assert len(domain_rows) == 1
    assert domain_rows[0]["mesh_element_types"] == "tri"
    assert domain_rows[0]["selected_as_vacuum_domain"] is True


def test_mesh_public_validation_errors_cover_each_owner_boundary(
    tmp_path: Path,
) -> None:
    mesh = _triangle_mesh()
    with pytest.raises(ValueError, match="positive finite value"):
        scale_mesh_coordinates(mesh, 0.0)
    with pytest.raises(ValueError, match="supports only 2D mesh"):
        build_precomputed_arrays(
            ParsedMesh(sdim=3, vertices=mesh.vertices, type_blocks=mesh.type_blocks),
            diagnostic_grid_spacing_m=0.5,
        )
    with pytest.raises(ValueError, match="must include tri or quad"):
        build_precomputed_arrays(
            ParsedMesh(sdim=2, vertices=mesh.vertices, type_blocks={}),
            diagnostic_grid_spacing_m=0.5,
        )
    with pytest.raises(ValueError, match=r"finite \(n, 2\) array"):
        domain_boundary_edge_vertex_ids(np.zeros((2, 3)), mesh.type_blocks)
    with pytest.raises(ValueError, match="must include edge entities"):
        assign_part_ids_from_edge_entities(
            {"tri": mesh.type_blocks["tri"]},
            np.asarray([[0, 1]], dtype=np.int64),
        )
    with pytest.raises(ValueError, match=r"must be an \(n, 2\)"):
        assign_part_ids_from_edge_entities(
            mesh.type_blocks,
            np.asarray([0, 1], dtype=np.int64),
        )
    invalid_edge_block = MeshTypeBlock(
        type_name="edg",
        vertices_per_element=3,
        elements=np.asarray([[0, 1, 2]], dtype=np.int64),
        geometric_entity_indices=np.asarray([0], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="exactly two mesh vertex IDs"):
        assign_part_ids_from_edge_entities(
            {"edg": invalid_edge_block},
            np.asarray([[0, 1]], dtype=np.int64),
        )

    assert (
        mesh_artifacts.write_comsol_entity_maps(
            tmp_path,
            ParsedMesh(sdim=2, vertices=mesh.vertices, type_blocks={}),
            active_part_ids=[],
            vacuum_domain_ids=(),
        )
        == {}
    )


def test_diagnostic_grid_rejects_invalid_or_excessive_axis_counts() -> None:
    with pytest.raises(ValueError, match="finite and strictly ordered"):
        mesh_topology._make_uniform_axis(1.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="positive and finite"):
        mesh_topology._make_uniform_axis(0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="too many points on one axis"):
        mesh_topology._make_uniform_axis(0.0, 1.0, 1.0e-9)

    mesh = _two_square_mesh(scale=1.0, separated=False)
    with pytest.raises(ValueError, match="too many 2D grid points"):
        build_precomputed_arrays(mesh, diagnostic_grid_spacing_m=2.0e-4)


def test_comsol_boundary_entity_rows_keep_schema_order_and_adjacency() -> None:
    source = _two_square_mesh(scale=1.0, separated=False)
    edge_block = source.type_blocks["edg"]
    quad_block = source.type_blocks["quad"]
    mesh = ParsedMesh(
        sdim=source.sdim,
        vertices=source.vertices,
        type_blocks={
            "edg": MeshTypeBlock(
                type_name=edge_block.type_name,
                vertices_per_element=edge_block.vertices_per_element,
                elements=edge_block.elements,
                geometric_entity_indices=np.asarray(
                    [2, 0, 2, 1, 0, 1, 3], dtype=np.int64
                ),
            ),
            "quad": MeshTypeBlock(
                type_name=quad_block.type_name,
                vertices_per_element=quad_block.vertices_per_element,
                elements=quad_block.elements,
                geometric_entity_indices=np.asarray([4, 1], dtype=np.int64),
            ),
        },
    )

    rows = mesh_artifacts._comsol_boundary_entity_rows(mesh, active_part_ids=[2])

    expected_columns = [
        "solver_part_id",
        "comsol_edge_entity_id",
        "raw_comsol_edge_entity_index",
        "comsol_api_selection_entity_id",
        "active_in_solver_boundary",
        "segment_count",
        "x_min_m",
        "x_max_m",
        "y_min_m",
        "y_max_m",
        "adjacent_domain_ids",
        "solver_part_name",
        "comsol_material_name",
    ]
    assert [list(row) for row in rows] == [expected_columns] * 4
    assert [
        (
            row["solver_part_id"],
            row["active_in_solver_boundary"],
            row["segment_count"],
            row["x_min_m"],
            row["x_max_m"],
            row["adjacent_domain_ids"],
        )
        for row in rows
    ] == [
        (1, False, 2, 0.0, 2.0, "2;5"),
        (2, True, 2, 0.0, 2.0, "2;5"),
        (3, False, 2, 0.0, 2.0, "2;5"),
        (4, False, 1, 1.0, 1.0, "2;5"),
    ]
    for part_id, row in enumerate(rows, start=1):
        assert row["comsol_edge_entity_id"] == part_id
        assert row["raw_comsol_edge_entity_index"] == part_id - 1
        assert row["comsol_api_selection_entity_id"] == part_id - 1
        assert row["y_min_m"] == 0.0
        assert row["y_max_m"] == 1.0
        assert row["solver_part_name"] == f"comsol_boundary_{part_id}"
        assert row["comsol_material_name"] == "not_exported_from_mphtxt"


def test_comsol_domain_rows_and_geometry_npz_keep_schema_and_dtypes(
    tmp_path: Path,
) -> None:
    source = _two_square_mesh(scale=1.0, separated=False)
    quad = source.type_blocks["quad"]
    mesh = ParsedMesh(
        sdim=source.sdim,
        vertices=source.vertices,
        type_blocks={
            **source.type_blocks,
            "quad": MeshTypeBlock(
                type_name=quad.type_name,
                vertices_per_element=quad.vertices_per_element,
                elements=quad.elements,
                geometric_entity_indices=np.asarray([0, 1], dtype=np.int64),
            ),
        },
    )
    rows = mesh_artifacts._comsol_domain_entity_rows(mesh, vacuum_domain_ids=(2,))

    expected_columns = [
        "comsol_domain_entity_id",
        "raw_comsol_domain_entity_index",
        "comsol_api_selection_entity_id",
        "selected_as_vacuum_domain",
        "element_count",
        "mesh_element_types",
        "x_min_m",
        "x_max_m",
        "y_min_m",
        "y_max_m",
        "comsol_material_name",
    ]
    assert [list(row) for row in rows] == [expected_columns, expected_columns]
    assert [row["comsol_domain_entity_id"] for row in rows] == [1, 2]
    assert [row["selected_as_vacuum_domain"] for row in rows] == [False, True]
    assert [row["mesh_element_types"] for row in rows] == ["quad", "quad"]

    arrays = build_precomputed_arrays(mesh, diagnostic_grid_spacing_m=0.5)
    output = tmp_path / "geometry.npz"
    write_geometry_npz(
        output,
        axes_x=np.asarray(arrays["axes_x"]),
        axes_y=np.asarray(arrays["axes_y"]),
        arrays=arrays,
        mesh=mesh,
        metadata={"coordinate_system": "cartesian_xy"},
    )
    with np.load(output) as payload:
        assert payload.files == [
            "axis_0",
            "axis_1",
            "sdf",
            "normal_0",
            "normal_1",
            "valid_mask",
            "nearest_boundary_part_id_map",
            "boundary_edges",
            "boundary_edge_part_ids",
            "boundary_loops_2d_flat",
            "boundary_loops_2d_offsets",
            "mesh_vertices",
            "mesh_triangles",
            "mesh_triangle_part_ids",
            "mesh_quads",
            "mesh_quad_part_ids",
            "metadata_json",
        ]
        assert payload["axis_0"].dtype == np.float64
        assert payload["sdf"].dtype == np.float64
        assert payload["valid_mask"].dtype == np.bool_
        assert payload["nearest_boundary_part_id_map"].dtype == np.int32
        assert payload["boundary_edge_part_ids"].dtype == np.int32
        assert payload["mesh_triangles"].dtype == np.int32
        assert payload["mesh_quads"].dtype == np.int32
        assert json.loads(str(payload["metadata_json"])) == {
            "coordinate_system": "cartesian_xy"
        }


def test_comsol_boundary_topology_errors_keep_actionable_context() -> None:
    mesh = _two_square_mesh(scale=1.0, separated=False)
    invalid_tri = MeshTypeBlock(
        type_name="tri",
        vertices_per_element=3,
        elements=np.asarray([[0, 1, 99]], dtype=np.int64),
        geometric_entity_indices=np.asarray([0], dtype=np.int64),
    )

    with pytest.raises(
        ValueError,
        match=(
            "COMSOL surface element references a mesh vertex outside the "
            "vertex table: edge=\\(1, 99\\), vertex_count=6"
        ),
    ):
        domain_boundary_edge_vertex_ids(mesh.vertices, {"tri": invalid_tri})

    with pytest.raises(
        ValueError,
        match=(
            "selected vacuum-domain boundary is missing an explicit COMSOL "
            "edge entity: edge=\\(0, 2\\)"
        ),
    ):
        assign_part_ids_from_edge_entities(
            mesh.type_blocks,
            np.asarray([[2, 0]], dtype=np.int64),
        )


def test_near_but_distinct_comsol_nodes_are_not_merged_by_decimal_rounding() -> None:
    mesh = _two_square_mesh(scale=1.0e-13, separated=True)

    boundary_vertex_ids = domain_boundary_edge_vertex_ids(
        mesh.vertices,
        mesh.type_blocks,
    )
    part_ids = assign_part_ids_from_edge_entities(mesh.type_blocks, boundary_vertex_ids)
    boundary_edges = mesh.vertices[boundary_vertex_ids]
    loops = build_boundary_loops_2d(boundary_edges)
    topology = validate_boundary_edges_2d(boundary_edges)

    assert boundary_vertex_ids.shape == (8, 2)
    assert part_ids.tolist() == list(range(1, 9))
    assert len(loops) == 2
    assert topology["vertex_count"] == 8
    assert topology["identity_resolution_m"] == pytest.approx(1.0e-13)
    assert topology["identity_tolerance_m"] < 0.02e-13


def _part_assignment_uses_node_ids_for_close_disconnected_edges() -> None:
    mesh = _two_square_mesh(scale=1.0e-13, separated=True)
    boundary_vertex_ids = domain_boundary_edge_vertex_ids(
        mesh.vertices,
        mesh.type_blocks,
    )

    # Nodes 1 and 4 are physically close but deliberately belong to different
    # disconnected boundaries and different COMSOL edge entities.
    assert np.linalg.norm(mesh.vertices[1] - mesh.vertices[4]) == pytest.approx(2.0e-15)
    part_ids = assign_part_ids_from_edge_entities(mesh.type_blocks, boundary_vertex_ids)
    keyed_parts = {
        tuple(sorted((int(edge[0]), int(edge[1])))): int(part_id)
        for edge, part_id in zip(boundary_vertex_ids, part_ids, strict=True)
    }
    assert keyed_parts[(0, 1)] == 1
    assert keyed_parts[(4, 5)] == 5


globals()[
    "test_part_assignment_uses_node_ids_when_an_edge_has_near_duplicate_coordinates"
] = _part_assignment_uses_node_ids_for_close_disconnected_edges
