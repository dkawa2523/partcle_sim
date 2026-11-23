from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from chamber_particles.case_format import (
    BoundaryData,
    DataBundle,
    FieldData,
    GeometryData,
    P1TriLayout,
    read,
    write,
)
from chamber_particles_tools.electrostatic_builder.numerics import (
    ELEMENTARY_CHARGE_C,
    ClosureParameters,
    SolverSettings,
    evaluate_closure,
    solve_axisymmetric_p1,
)
from chamber_particles_tools.electrostatic_builder.workflow import (
    build_from_configuration,
    check_from_configuration,
)


def _f8(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype="<f8")


def _i8(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype="<i8")


def _i4(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype="<i4")


def _u1(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype="<u1")


def _i1(value: object) -> np.ndarray:
    return np.ascontiguousarray(value, dtype="<i1")


def _closure(**changes: float) -> ClosureParameters:
    values = {
        "bulk_number_density_m3": 1.0e14,
        "electron_temperature_V": 4.0,
        "positive_ion_mass_kg": 50.0e-3 / 6.02214076e23,
        "bulk_potential_V": 0.0,
        "sheath_smoothing_V": 0.05,
        "density_floor_m3": 1.0e6,
        "ion_speed_floor_m_s": 1.0e-3,
    }
    values.update(changes)
    return ClosureParameters(**values)


def _settings() -> SolverSettings:
    return SolverSettings(
        continuation_ramps=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0),
        max_newton_iterations=30,
        relative_residual_tolerance=1.0e-9,
        absolute_residual_tolerance_C=1.0e-24,
        max_linear_iterations=500,
        linear_krylov_dimension=30,
        linear_relative_tolerance=1.0e-11,
        minimum_line_search_factor=1.0 / 1024.0,
    )


def _grid(
    radial_count: int,
    axial_count: int,
    *,
    radial_extent: tuple[float, float] = (0.0, 0.02),
    axial_extent: tuple[float, float] = (0.0, 0.02),
) -> tuple[np.ndarray, np.ndarray]:
    radial = np.linspace(*radial_extent, radial_count)
    axial = np.linspace(*axial_extent, axial_count)
    nodes = _f8([[r, z] for r in radial for z in axial])
    cells: list[list[int]] = []
    for radial_index in range(radial_count - 1):
        for axial_index in range(axial_count - 1):
            lower_left = radial_index * axial_count + axial_index
            lower_right = (radial_index + 1) * axial_count + axial_index
            upper_right = lower_right + 1
            upper_left = lower_left + 1
            cells.extend(
                ([lower_left, lower_right, upper_right], [lower_left, upper_right, upper_left])
            )
    return nodes, _i8(cells)


def test_closure_is_quasineutral_at_bulk_and_derivative_matches_finite_difference() -> None:
    parameters = _closure()
    potential = _f8([-8.0, -1.0, -0.02, 0.0, 0.02, 0.2])
    state = evaluate_closure(potential, parameters)

    bulk = 3
    bohm_speed = math.sqrt(
        ELEMENTARY_CHARGE_C * parameters.electron_temperature_V / parameters.positive_ion_mass_kg
    )
    assert state.electron_number_density_m3[bulk] == parameters.bulk_number_density_m3
    assert state.positive_ion_number_density_m3[bulk] == parameters.bulk_number_density_m3
    assert state.space_charge_density_C_m3[bulk] == 0.0
    assert state.positive_ion_speed_m_s[bulk] == bohm_speed

    step = 1.0e-6
    upper = evaluate_closure(_f8(potential + step), parameters).space_charge_density_C_m3
    lower = evaluate_closure(_f8(potential - step), parameters).space_charge_density_C_m3
    finite_difference = (upper - lower) / (2.0 * step)
    assert np.allclose(
        state.charge_derivative_C_m3_V,
        finite_difference,
        rtol=2.0e-7,
        atol=2.0e-12,
    )


def test_axisymmetric_laplace_affine_solution_and_electric_field() -> None:
    nodes, cells = _grid(6, 7, radial_extent=(0.0, 0.03), axial_extent=(0.0, 0.02))
    on_boundary = (nodes[:, 0] == 0.03) | (nodes[:, 1] == 0.0) | (nodes[:, 1] == 0.02)
    fixed = np.flatnonzero(on_boundary).astype("<i8", copy=False)
    expected = 2.0 + 30.0 * nodes[:, 1]
    closure = _closure(bulk_potential_V=2.0, sheath_smoothing_V=1.0e-5)

    solution = solve_axisymmetric_p1(
        nodes,
        cells,
        fixed,
        _f8(expected[fixed]),
        closure,
        _settings(),
    )

    assert np.allclose(solution.potential_V, expected, rtol=0.0, atol=2.0e-10)
    assert np.allclose(solution.electric_field_V_m[:, 0], 0.0, rtol=0.0, atol=2.0e-10)
    assert np.allclose(solution.electric_field_V_m[:, 1], -30.0, rtol=0.0, atol=2.0e-9)
    assert solution.report.relative_residual <= 1.0e-9
    assert solution.report.charge_balance_error_C <= 2.0e-17


def test_axisymmetric_annular_laplace_converges_to_logarithmic_solution() -> None:
    errors: list[float] = []
    inner_radius = 0.01
    outer_radius = 0.03
    for count in (5, 9, 17):
        nodes, cells = _grid(
            count,
            count,
            radial_extent=(inner_radius, outer_radius),
            axial_extent=(0.0, 0.01),
        )
        fixed_mask = (nodes[:, 0] == inner_radius) | (nodes[:, 0] == outer_radius)
        fixed = np.flatnonzero(fixed_mask).astype("<i8", copy=False)
        expected = np.log(nodes[:, 0] / inner_radius) / math.log(outer_radius / inner_radius)
        solution = solve_axisymmetric_p1(
            nodes,
            cells,
            fixed,
            _f8(expected[fixed]),
            _closure(bulk_potential_V=-1.0, sheath_smoothing_V=1.0e-5),
            _settings(),
        )
        errors.append(float(np.max(np.abs(solution.potential_V - expected))))

    assert errors[1] < 0.35 * errors[0]
    assert errors[2] < 0.35 * errors[1]


def test_disconnected_component_without_potential_anchor_is_rejected() -> None:
    first_nodes, first_cells = _grid(3, 3)
    second_nodes = first_nodes + _f8([0.0, 0.04])
    nodes = _f8(np.concatenate((first_nodes, second_nodes)))
    cells = _i8(np.concatenate((first_cells, first_cells + first_nodes.shape[0])))

    with pytest.raises(ValueError, match=r"every connected mesh component.*Dirichlet"):
        solve_axisymmetric_p1(nodes, cells, _i8([0]), _f8([0.0]), _closure(), _settings())


def test_disconnected_anchored_components_recover_independent_affine_fields() -> None:
    first_nodes, first_cells = _grid(3, 3)
    second_nodes = first_nodes + _f8([0.0, 0.04])
    nodes = _f8(np.concatenate((first_nodes, second_nodes)))
    cells = _i8(np.concatenate((first_cells, first_cells + first_nodes.shape[0])))
    boundary = (first_nodes[:, 0] == 0.02) | (first_nodes[:, 1] != 0.01)
    fixed = np.flatnonzero(np.tile(boundary, 2)).astype("<i8", copy=False)
    expected = _f8(
        np.concatenate((2.0 + 30.0 * first_nodes[:, 1], 4.0 + 20.0 * second_nodes[:, 1]))
    )

    solution = solve_axisymmetric_p1(
        nodes, cells, fixed, _f8(expected[fixed]), _closure(), _settings()
    )

    np.testing.assert_allclose(solution.potential_V, expected, rtol=0.0, atol=2.0e-10)
    np.testing.assert_allclose(solution.electric_field_V_m[:, 0], 0.0, atol=2.0e-10)
    np.testing.assert_allclose(solution.electric_field_V_m[:9, 1], -30.0, atol=2.0e-9)
    np.testing.assert_allclose(solution.electric_field_V_m[9:, 1], -20.0, atol=2.0e-9)


def test_closure_rejects_nonfinite_derived_quantities() -> None:
    with pytest.raises(ValueError, match="non-finite derived quantity"):
        evaluate_closure(_f8([-np.finfo(np.float64).max]), _closure())


def test_nonlinear_sheath_converges_with_positive_space_charge() -> None:
    nodes, cells = _grid(7, 25, radial_extent=(0.0, 0.01), axial_extent=(0.0, 0.01))
    fixed_mask = (nodes[:, 1] == 0.0) | (nodes[:, 1] == 0.01)
    fixed = np.flatnonzero(fixed_mask).astype("<i8", copy=False)
    target = np.where(nodes[fixed, 1] == 0.0, -8.0, 0.0)

    solution = solve_axisymmetric_p1(
        nodes,
        cells,
        fixed,
        _f8(target),
        _closure(),
        _settings(),
    )

    assert solution.report.relative_residual <= 1.0e-9
    assert solution.report.charge_balance_error_C <= 5.0e-17
    assert float(np.min(solution.potential_V)) == -8.0
    assert float(np.max(solution.potential_V)) <= 1.0e-12
    assert float(np.max(solution.space_charge_density_C_m3)) > 0.0
    assert np.all(solution.electron_number_density_m3 > 0.0)
    assert np.all(solution.positive_ion_number_density_m3 > 0.0)
    assert np.all(solution.electric_field_V_m[nodes[:, 0] == 0.0, 0] == 0.0)


def test_nonlinear_sheath_has_nested_mesh_self_convergence() -> None:
    profiles = [_nonlinear_profile(count) for count in (5, 9, 17)]
    coarse_error = float(np.linalg.norm(profiles[0] - profiles[2][::4]))
    medium_error = float(np.linalg.norm(profiles[1] - profiles[2][::2]))

    assert medium_error < 0.45 * coarse_error


def test_canonical_workflow_preserves_input_and_adds_particle_ready_fields(
    tmp_path: Path,
) -> None:
    source = tmp_path / "thermal.h5"
    output = tmp_path / "augmented.h5"
    report = output / "report.json"
    write(source, _thermal_bundle())
    config = tmp_path / "builder.yaml"
    config.write_text(_configuration_text(source.name), encoding="utf-8")

    checked = check_from_configuration(config)
    assert checked["status"] == "checked"
    assert not output.exists()
    summary = build_from_configuration(config, output)
    built = read(output / "bundle.h5")
    fields = {field.name: field for field in built.fields}

    assert summary["status"] == "complete"
    assert report.exists()
    assert fields["gas_velocity"].unit == "m/s"
    assert fields["electric_field"].components == ("r", "z")
    assert fields["electron_temperature"].unit == "K"
    assert fields["positive_ion_velocity"].stored_basis == "axisymmetric_rz"
    assert fields["positive_ion_mass"].unit == "kg"
    assert len(fields) == 12
    provenance = json.loads(built.provenance_json)
    assert provenance["producer"] == "chamber_particles_tools.electrostatic_builder"
    assert provenance["producer_metadata"]["model_revision"] == ("boltzmann_bohm_sheath_c2_v1")
    assert provenance["producer_metadata"]["boundary_conditions"][0]["group"] == "wafer"


def test_cli_reports_nonlinear_nonconvergence_without_publishing(tmp_path: Path) -> None:
    data = _thermal_bundle()
    nodes = _f8([*data.geometry.nodes_m, [0.01, 0.01]])
    cells = _i8([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    geometry = replace(
        data.geometry,
        nodes_m=nodes,
        tri3=cells,
        tri3_domain_id=_i4([3] * 4),
        boundary=replace(data.geometry.boundary, owner_cell_local_index=_i8([0, 1, 2, 3])),
    )
    fields = tuple(
        replace(field, values=_f8(np.vstack((field.values, field.values.mean(axis=0)))))
        for field in data.fields
    )
    write(
        tmp_path / "thermal.h5",
        replace(
            data,
            geometry=geometry,
            layouts=(P1TriLayout("plasma", nodes, cells, _u1([1] * 4)),),
            fields=fields,
        ),
    )
    configuration = tmp_path / "builder.yaml"
    configuration.write_text(
        _configuration_text("thermal.h5")
        .replace("max_newton_iterations: 30", "max_newton_iterations: 1")
        .replace(
            "absolute_residual_tolerance_C: 1.0e-18", "absolute_residual_tolerance_C: 1.0e-28"
        ),
        encoding="utf-8",
    )
    output = tmp_path / "not-accepted.h5"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "chamber_particles",
            "inputs",
            "electrostatic",
            "build",
            str(configuration),
            str(output),
            "--json",
        ],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2 and not result.stdout
    payload = json.loads(result.stderr)
    assert payload["error_type"] == "NonlinearSolveError"
    assert not output.exists()


@pytest.mark.parametrize("missing_facet", [False, True])
def test_builder_rejects_unprescribed_perimeter_even_when_all_nodes_are_fixed(
    tmp_path: Path, missing_facet: bool
) -> None:
    source = tmp_path / "thermal.h5"
    data = _thermal_bundle()
    configuration = _configuration_text(source.name)
    if missing_facet:
        boundary = data.geometry.boundary
        retained = [0, 2, 3]
        boundary = replace(
            boundary,
            line2=_i8(boundary.line2[retained]),
            boundary_id=_i4(boundary.boundary_id[retained]),
            group_id=_i4([0, 1, 0]),  # Bottom and omitted right edge share the wall group.
            material_id=_i4(boundary.material_id[retained]),
            owner_cell_type=_u1(boundary.owner_cell_type[retained]),
            owner_cell_local_index=_i8(boundary.owner_cell_local_index[retained]),
            orientation=_i1(boundary.orientation[retained]),
        )
        data = replace(
            data,
            geometry=replace(data.geometry, boundary=boundary, group_names=("wall", "bulk")),
        )
        configuration = configuration.replace(
            "  - group: wafer\n    priority: 20\n    potential: {kind: constant, value_V: -20.0}\n",
            "",
        )
    else:
        configuration = configuration.replace(
            "  - group: wall\n    priority: 10\n    potential: {kind: constant, value_V: 0.0}\n",
            "",
        )
    write(source, data)
    config = tmp_path / "builder.yaml"
    config.write_text(
        configuration,
        encoding="utf-8",
    )
    output = tmp_path / "augmented.h5"
    report = output / "report.json"

    with pytest.raises(ValueError, match=r"non-axis perimeter.*potential"):
        build_from_configuration(config, output)

    assert not output.exists()
    assert not report.exists()


def test_builder_axis_facets_do_not_impose_a_material_groups_potential(tmp_path: Path) -> None:
    original = _thermal_bundle()
    nodes, cells = _grid(2, 3)
    boundary = BoundaryData(
        line2=_i8([[0, 3], [3, 4], [4, 5], [5, 2], [2, 1], [1, 0]]),
        boundary_id=_i4([10, 20, 20, 30, 20, 20]),
        group_id=_i4([0, 1, 1, 2, 1, 1]),
        material_id=_i4([1] * 6),
        owner_cell_type=_u1([1] * 6),
        owner_cell_local_index=_i8([0, 0, 2, 3, 3, 1]),
        orientation=_i1([1] * 6),
    )
    geometry = replace(
        original.geometry, nodes_m=nodes, tri3=cells, tri3_domain_id=_i4([3] * 4), boundary=boundary
    )
    data = replace(
        original,
        geometry=geometry,
        layouts=(P1TriLayout("plasma", nodes, cells, _u1([1] * 4)),),
        fields=tuple(
            replace(field, values=_f8(np.tile(field.values[0], (6, 1))))
            for field in original.fields
        ),
    )
    potentials = []
    for include_axis in (True, False):
        source = tmp_path / f"thermal-{include_axis}.h5"
        write(source, data)
        configuration = tmp_path / f"builder-{include_axis}.yaml"
        configuration.write_text(
            _configuration_text(source.name)
            .replace("value_V: -20.0", "value_V: 15.0")
            .replace("value_V: 0.0", "value_V: 20.0"),
            encoding="utf-8",
        )
        output = tmp_path / f"augmented-{include_axis}.h5"
        build_from_configuration(configuration, output)
        built = read(output / "bundle.h5")
        potentials.append(
            next(field.values for field in built.fields if field.name == "electric_potential")
        )
        boundary = replace(
            boundary,
            line2=_i8(boundary.line2[:4]),
            boundary_id=_i4(boundary.boundary_id[:4]),
            group_id=_i4(boundary.group_id[:4]),
            material_id=_i4(boundary.material_id[:4]),
            owner_cell_type=_u1(boundary.owner_cell_type[:4]),
            owner_cell_local_index=_i8(boundary.owner_cell_local_index[:4]),
            orientation=_i1(boundary.orientation[:4]),
        )
        data = replace(data, geometry=replace(geometry, boundary=boundary))

    np.testing.assert_array_equal(potentials[0], potentials[1])
    assert 15.0 < potentials[0][1, 0] < 20.0


@pytest.mark.parametrize(
    ("replacement", "error"),
    [
        ("  electron_temperature_V: 4.0\n  electron_temperature_V: 5.0", "duplicate YAML key"),
        ("  <<: {electron_temperature_V: 4.0}\n  electron_temperature_V: 4.0", "merge keys"),
        (f"  electron_temperature_V: {10**400}", "must be a finite number"),
    ],
)
def test_builder_rejects_ambiguous_or_unrepresentable_input_before_publication(
    tmp_path: Path, replacement: str, error: str
) -> None:
    config = tmp_path / "builder.yaml"
    config.write_text(
        _configuration_text("missing-source.h5").replace(
            "  electron_temperature_V: 4.0", replacement
        ),
        encoding="utf-8",
    )
    output = tmp_path / "augmented.h5"
    with pytest.raises(ValueError, match=error):
        build_from_configuration(config, output)
    assert not output.exists()


def _nonlinear_profile(mesh_count: int) -> np.ndarray:
    nodes, cells = _grid(
        mesh_count,
        mesh_count,
        radial_extent=(0.0, 0.006),
        axial_extent=(0.0, 0.01),
    )
    fixed_mask = (nodes[:, 1] == 0.0) | (nodes[:, 1] == 0.01)
    fixed = np.flatnonzero(fixed_mask).astype("<i8", copy=False)
    target = np.where(nodes[fixed, 1] == 0.0, -8.0, 0.0)
    solution = solve_axisymmetric_p1(
        nodes,
        cells,
        fixed,
        _f8(target),
        _closure(),
        _settings(),
    )
    start = (mesh_count // 2) * mesh_count
    return solution.potential_V[start : start + mesh_count]


def _thermal_bundle() -> DataBundle:
    nodes = _f8([[0.0, 0.0], [0.02, 0.0], [0.02, 0.02], [0.0, 0.02]])
    cells = _i8([[0, 1, 2], [0, 2, 3]])
    boundary = BoundaryData(
        line2=_i8([[0, 1], [2, 1], [2, 3], [3, 0]]),
        boundary_id=_i4([10, 20, 30, 20]),
        group_id=_i4([0, 1, 2, 1]),
        material_id=_i4([1, 1, 1, 1]),
        owner_cell_type=_u1([1, 1, 1, 1]),
        owner_cell_local_index=_i8([0, 0, 1, 1]),
        orientation=_i1([1, -1, 1, 1]),
    )
    geometry = GeometryData(
        nodes_m=nodes,
        boundary=boundary,
        group_names=("wafer", "wall", "bulk"),
        tri3=cells,
        tri3_domain_id=_i4([3, 3]),
    )
    layout = P1TriLayout("plasma", nodes, cells, _u1([1, 1]))
    velocity = FieldData(
        "gas_velocity",
        "plasma",
        "node",
        ("r", "z"),
        "axisymmetric_rz",
        _f8([[0.0, 0.1], [0.0, 0.1], [0.0, 0.1], [0.0, 0.1]]),
        "m/s",
    )
    temperature = FieldData(
        "gas_temperature",
        "plasma",
        "node",
        ("value",),
        "scalar",
        _f8([[300.0], [310.0], [320.0], [305.0]]),
        "K",
    )
    provenance = json.dumps(
        {
            "producer": "unit_test",
            "producer_version": "1",
            "source_sha256": f"sha256:{'1' * 64}",
            "field_semantics_revision": "thermal_v1",
            "producer_metadata": {},
        }
    )
    return DataBundle(
        "axisymmetric_rz", provenance, geometry, (layout,), (velocity, temperature), ()
    )


def _configuration_text(source_name: str) -> str:
    return f"""\
format_version: 2
input:
  data_path: {source_name}
  layout: plasma
  gas_velocity_field: gas_velocity
  gas_temperature_field: gas_temperature
model:
  revision: boltzmann_bohm_sheath_c2_v1
  bulk_number_density_m3: 1.0e+14
  electron_temperature_V: 4.0
  positive_ion_mass_kg: 8.302695335869234e-26
  bulk_potential_V: 15.0
  ion_mobility_m2_V_s: 1.0
  ion_flux_regularization_speed_m_s: 1.0
  sheath_smoothing_V: 0.05
  density_floor_m3: 1.0e+6
  ion_speed_floor_m_s: 1.0e-3
boundaries:
  - group: wafer
    priority: 20
    potential: {{kind: constant, value_V: -20.0}}
  - group: wall
    priority: 10
    potential: {{kind: constant, value_V: 0.0}}
  - group: bulk
    priority: 30
    potential: {{kind: constant, value_V: 15.0}}
solver:
  continuation_ramps: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
  max_newton_iterations: 30
  relative_residual_tolerance: 1.0e-9
  absolute_residual_tolerance_C: 1.0e-18
  max_linear_iterations: 200
  linear_krylov_dimension: 20
  linear_relative_tolerance: 1.0e-11
  minimum_line_search_factor: 0.0009765625
"""
