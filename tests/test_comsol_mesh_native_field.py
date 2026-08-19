"""Mesh-native COMSOL field artifacts keep support on the solved mesh."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from particle_tracer_unified.comsol_case.builder import write_case_files
from particle_tracer_unified.comsol_case.fields import pack_mesh_field_bundle
from particle_tracer_unified.comsol_case.mesh import (
    parse_comsol_mphtxt,
    scale_mesh_coordinates,
    select_vacuum_domains,
)
from particle_tracer_unified.core.field_sampling import (
    VALID_MASK_STATUS_CLEAN,
    VALID_MASK_STATUS_HARD_INVALID,
)
from particle_tracer_unified.core.triangle_mesh_sampling_2d import (
    sample_triangle_mesh_series,
    sample_triangle_mesh_status,
)
from particle_tracer_unified.io.comsol_manifest import ComsolCaseManifest
from particle_tracer_unified.providers.precomputed import (
    build_precomputed_triangle_mesh_field,
)

# A graded mesh: the column of nodes near x = 0 is refined the way a COMSOL
# boundary layer is, which is exactly the structure a uniform resample loses.
_MPHTXT = """2 # sdim
6 # number of mesh vertices
# Mesh vertex coordinates
0 0
0.01 0
1 0
0 1
0.01 1
1 1
2 # number of element types
3 edg # type name
2 # number of vertices per element
6 # number of elements
# Elements
0 1
1 2
2 5
5 4
4 3
3 0
6 # number of geometric entity indices
# Geometric entity indices
0
0
1
2
2
3
4 quad # type name
4 # number of vertices per element
2 # number of elements
# Elements
0 1 4 3
1 2 5 4
2 # number of geometric entity indices
# Geometric entity indices
0
0
"""

_NODE_VALUES = {
    # A steep near-wall ramp in the first 10 mm, flat afterwards: the shape a
    # sheath field has and a 1 m grid cell cannot represent.
    "ux": [0.0, 100.0, 100.0, 0.0, 100.0, 100.0],
    "uy": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "mu": [1.8e-5] * 6,
}


def _write_mesh(path: Path) -> Path:
    path.write_text(_MPHTXT, encoding="utf-8")
    return path


def _write_node_samples(path: Path) -> Path:
    pd.DataFrame({"node_index": list(range(6)), **_NODE_VALUES}).to_csv(
        path, index=False
    )
    return path


def _write_release(path: Path) -> Path:
    pd.DataFrame(
        {
            "particle_id": [1],
            "x_m": [0.5],
            "y_m": [0.5],
            "vx_mps": [0.0],
            "vy_mps": [0.0],
            "release_time_s": [0.0],
            "mass_kg": [1.0e-15],
            "drag_diameter_m": [1.0e-6],
            "charge_C": [0.0],
            "source_part_id": [1],
        }
    ).to_csv(path, index=False)
    return path


def _write_boundaries(path: Path) -> Path:
    pd.DataFrame(
        {
            "part_id": [1, 2, 3, 4],
            "comsol_entity_id": [1, 2, 3, 4],
            "part_name": ["b1", "b2", "b3", "b4"],
            "role": ["wall"] * 4,
            "material_id": [1] * 4,
            "material_name": ["steel"] * 4,
            "wall_law": ["specular"] * 4,
            "wall_stick_probability": [0.0] * 4,
            "wall_restitution": [1.0] * 4,
            "wall_diffuse_fraction": [0.0] * 4,
            "wall_critical_sticking_velocity_mps": [0.0] * 4,
            "metadata_json": ["{}"] * 4,
        }
    ).to_csv(path, index=False)
    return path


def _selected_mesh(mesh_path: Path):
    parsed = scale_mesh_coordinates(parse_comsol_mphtxt(mesh_path), 1.0)
    mesh, _domains = select_vacuum_domains(parsed, (1,))
    return mesh


def test_mesh_bundle_keeps_every_solved_vertex(tmp_path: Path) -> None:
    mesh_path = _write_mesh(tmp_path / "mesh.mphtxt")
    samples = _write_node_samples(tmp_path / "field_samples_nodes.csv")

    packed = pack_mesh_field_bundle(
        samples,
        tmp_path / "field.npz",
        mesh=_selected_mesh(mesh_path),
    )

    assert packed.summary["mode"] == "mesh_native"
    assert packed.summary["mesh_vertex_count"] == 6
    assert packed.summary["mesh_triangle_count"] == 4
    # The graded column survives: the shortest edge is the 10 mm near-wall
    # spacing, not an averaged cell size.
    assert packed.summary["min_edge_length_m"] == pytest.approx(0.01)
    with np.load(packed.path) as payload:
        assert "valid_mask" not in payload.files
        assert payload["mesh_vertices"].shape == (6, 2)
        np.testing.assert_allclose(payload["ux"], _NODE_VALUES["ux"])
        metadata = json.loads(str(np.asarray(payload["metadata_json"]).item()))
    assert metadata["node_identity"] == "comsol_mphtxt_global_vertex_index"


def test_mesh_bundle_rejects_missing_or_nonfinite_nodes(tmp_path: Path) -> None:
    mesh_path = _write_mesh(tmp_path / "mesh.mphtxt")
    mesh = _selected_mesh(mesh_path)

    partial = tmp_path / "partial.csv"
    pd.DataFrame(
        {"node_index": [0, 1, 2], "ux": [0.0, 1.0, 2.0]},
    ).to_csv(partial, index=False)
    with pytest.raises(ValueError, match="missing mesh vertices"):
        pack_mesh_field_bundle(partial, tmp_path / "a.npz", mesh=mesh)

    holed = tmp_path / "holed.csv"
    values = dict(_NODE_VALUES)
    values["ux"] = [0.0, float("nan"), 100.0, 0.0, 100.0, 100.0]
    pd.DataFrame({"node_index": list(range(6)), **values}).to_csv(holed, index=False)
    with pytest.raises(ValueError, match="non-finite at mesh vertices"):
        pack_mesh_field_bundle(holed, tmp_path / "b.npz", mesh=mesh)

    grid_table = tmp_path / "grid.csv"
    pd.DataFrame({"x": [0.0], "y": [0.0], "ux": [1.0]}).to_csv(grid_table, index=False)
    with pytest.raises(ValueError, match="node_index"):
        pack_mesh_field_bundle(grid_table, tmp_path / "c.npz", mesh=mesh)


def test_mesh_field_resolves_the_near_wall_ramp_a_coarse_grid_cannot(
    tmp_path: Path,
) -> None:
    mesh_path = _write_mesh(tmp_path / "mesh.mphtxt")
    samples = _write_node_samples(tmp_path / "nodes.csv")
    packed = pack_mesh_field_bundle(
        samples, tmp_path / "field.npz", mesh=_selected_mesh(mesh_path)
    )
    provider = build_precomputed_triangle_mesh_field(
        {"npz_path": str(packed.path)},
        2,
        "cartesian_xy",
    )
    field = provider.field
    series = field.quantities["ux"]

    # Halfway across the refined 10 mm column the exact ramp value is 50 m/s.
    midpoint = sample_triangle_mesh_series(series, field, np.asarray([0.005, 0.5]), 0.0)
    assert midpoint == pytest.approx(50.0, rel=1.0e-12)
    # Outside the refined column the field is flat.
    assert sample_triangle_mesh_series(
        series, field, np.asarray([0.5, 0.5]), 0.0
    ) == pytest.approx(100.0)

    # Support ends with the mesh: a point past the wall is not supported, even
    # though a value query there still clamps to the nearest element so a trial
    # step that crosses the wall stays finite for hit localization.
    outside = np.asarray([1.5, 0.5])
    assert sample_triangle_mesh_status(field, outside) == int(
        VALID_MASK_STATUS_HARD_INVALID
    )
    assert sample_triangle_mesh_series(series, field, outside, 0.0) == pytest.approx(
        100.0
    )
    assert sample_triangle_mesh_status(field, np.asarray([0.5, 0.5])) == int(
        VALID_MASK_STATUS_CLEAN
    )


def test_builder_writes_a_triangle_mesh_field_artifact(tmp_path: Path) -> None:
    mesh_path = _write_mesh(tmp_path / "mesh.mphtxt")
    samples = _write_node_samples(tmp_path / "field_samples_nodes.csv")
    out = tmp_path / "case"

    write_case_files(
        mesh_path,
        out,
        field_node_samples_path=samples,
        release_table_path=_write_release(tmp_path / "release.csv"),
        boundaries_path=_write_boundaries(tmp_path / "walls.csv"),
        diagnostic_grid_spacing_m=0.25,
        coordinate_scale_m_per_model_unit=1.0,
        coordinate_system="cartesian_xy",
        model_name="graded-test",
        study="std1",
        dataset="dset1",
        solution="sol1",
        solution_number=1,
        vacuum_domain_ids=(1,),
        drag_law="stokes",
        gas_dynamic_viscosity_Pas=1.8e-5,
        solver_dt_s=0.1,
        solver_t_end_s=1.0,
    )

    manifest = ComsolCaseManifest.load(out / "comsol_manifest.yaml")
    manifest.validate(strict=True)
    assert manifest.artifacts["field"].format == "precomputed_triangle_mesh_npz"
    assert manifest.metadata["field_storage"] == "mesh_native"
    # The runtime provider kind follows the declared artifact format, so the
    # solver samples the mesh rather than a resampled lattice.
    assert manifest.provider_config()["field"]["kind"] == (
        "precomputed_triangle_mesh_npz"
    )

    summary = json.loads(
        (out / "generated" / "comsol_case_summary.json").read_text(encoding="utf-8")
    )
    assert summary["field_mode"] == "mesh_native"
    assert summary["field_summary"]["mesh_triangle_count"] == 4
    assert summary["field_summary"]["min_edge_length_m"] == pytest.approx(0.01)


def test_builder_rejects_two_declared_field_sources(tmp_path: Path) -> None:
    mesh_path = _write_mesh(tmp_path / "mesh.mphtxt")
    samples = _write_node_samples(tmp_path / "nodes.csv")
    bundle = tmp_path / "grid.npz"
    np.savez(bundle, axis_0=np.zeros(2), axis_1=np.zeros(2))

    with pytest.raises(ValueError, match="exactly one field source"):
        write_case_files(
            mesh_path,
            tmp_path / "case",
            field_bundle_path=bundle,
            field_node_samples_path=samples,
            diagnostic_grid_spacing_m=0.25,
            coordinate_scale_m_per_model_unit=1.0,
            vacuum_domain_ids=(1,),
        )


def _write_uniform_node_samples(path: Path, ux: float) -> Path:
    pd.DataFrame(
        {
            "node_index": list(range(6)),
            "ux": [ux] * 6,
            "uy": [0.0] * 6,
            "mu": [1.8e-5] * 6,
            "rho": [1.2] * 6,
            "T": [300.0] * 6,
        }
    ).to_csv(path, index=False)
    return path


def _write_sticking_boundaries(path: Path) -> Path:
    frame = pd.read_csv(_write_boundaries(path))
    frame.loc[frame["part_id"] == 4, "wall_law"] = "stick"
    frame.to_csv(path, index=False)
    return path


def test_a_mesh_native_case_carries_a_particle_all_the_way_to_the_wall(
    tmp_path: Path,
) -> None:
    """The wall, not a support hole, must terminate a particle.

    A resampled lattice leaves every boundary lined with cells whose stencil
    touches a node outside the vacuum domain.  A particle entering that band
    stops with ``invalid_mask_stopped`` short of the wall, while COMSOL runs it
    into the wall condition.  Mesh-native support ends exactly at the domain
    boundary, so the wall law is what decides the outcome.
    """

    from particle_tracer_unified import load_case, simulate

    out = tmp_path / "case"
    write_case_files(
        _write_mesh(tmp_path / "mesh.mphtxt"),
        out,
        field_node_samples_path=_write_uniform_node_samples(
            tmp_path / "field_samples_nodes.csv", -1.0
        ),
        release_table_path=_write_release(tmp_path / "release.csv"),
        boundaries_path=_write_sticking_boundaries(tmp_path / "walls.csv"),
        diagnostic_grid_spacing_m=0.25,
        coordinate_scale_m_per_model_unit=1.0,
        coordinate_system="cartesian_xy",
        model_name="wall-reach",
        study="std1",
        dataset="dset1",
        solution="sol1",
        solution_number=1,
        vacuum_domain_ids=(1,),
        drag_law="stokes",
        gas_dynamic_viscosity_Pas=1.8e-5,
        solver_dt_s=0.05,
        solver_t_end_s=0.6,
    )

    result = simulate(load_case(out / "run_config.yaml"))
    state = result.state

    assert [str(value) for value in state.terminal_state] == ["stuck"]
    # Released at x = 0.5 and carried by a -1 m/s flow into the x = 0 wall,
    # which is where the particle must end up rather than one cell short of it.
    assert float(state.position_m[0, 0]) == pytest.approx(0.0, abs=1.0e-9)
    assert result.stats.terminal_counts.get("stuck", 0) == 1


def _write_boundary_release(path: Path) -> Path:
    """Release exactly on the x = 0 wall (part 4), moving into the domain."""

    pd.DataFrame(
        {
            "particle_id": [1],
            "x_m": [0.0],
            "y_m": [0.5],
            "vx_mps": [0.0],
            "vy_mps": [0.0],
            "release_time_s": [0.0],
            "mass_kg": [1.0e-15],
            "drag_diameter_m": [1.0e-6],
            "charge_C": [0.0],
            "source_part_id": [4],
        }
    ).to_csv(path, index=False)
    return path


def test_release_on_its_own_boundary_is_accepted_and_moves_inward(
    tmp_path: Path,
) -> None:
    """A particle released on its declared entity behaves like a COMSOL inlet.

    COMSOL releases inlet particles on the boundary itself, where the inlet
    feature overrides the wall condition.  Preflight must accept that position
    instead of demanding an artificial inward displacement, and the first step
    must carry the particle into the domain rather than register a hit on the
    surface it just left.
    """

    from particle_tracer_unified import load_case, simulate, validate_case

    out = tmp_path / "case"
    write_case_files(
        _write_mesh(tmp_path / "mesh.mphtxt"),
        out,
        field_node_samples_path=_write_uniform_node_samples(
            tmp_path / "field_samples_nodes.csv", 1.0
        ),
        release_table_path=_write_boundary_release(tmp_path / "release.csv"),
        boundaries_path=_write_boundaries(tmp_path / "walls.csv"),
        diagnostic_grid_spacing_m=0.25,
        coordinate_scale_m_per_model_unit=1.0,
        coordinate_system="cartesian_xy",
        model_name="inlet-release",
        release_projection_tolerance_m=1.0e-9,
        gas_temperature_K=300.0,
        gas_density_kgm3=1.2,
        gas_molecular_mass_amu=39.948,
        study="std1",
        dataset="dset1",
        solution="sol1",
        solution_number=1,
        vacuum_domain_ids=(1,),
        drag_law="stokes",
        gas_dynamic_viscosity_Pas=1.8e-5,
        solver_dt_s=0.05,
        solver_t_end_s=0.3,
    )

    case = load_case(out / "run_config.yaml")
    report = validate_case(case, detail="full")
    assert report.passed, report

    result = simulate(case)
    state = result.state
    # Carried inward by the +1 m/s flow, with no wall event on the release
    # surface it started on.
    assert float(state.position_m[0, 0]) > 0.25
    assert dict(result.wall_summary) == {}
    assert [str(value) for value in state.terminal_state] == ["active_free_flight"]
