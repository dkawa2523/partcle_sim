from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from chamber_particles import load_case
from chamber_particles.physics.catalog import resolve_physics_plan
from tests.verification.microcases import MICROCASE_IDS, materialize_microcase


@pytest.mark.parametrize("case_id", MICROCASE_IDS)
def test_microcase_materializes_through_canonical_writer_and_public_loader(
    tmp_path: Path, case_id: str
) -> None:
    materialized = materialize_microcase(case_id, tmp_path / case_id)

    case = load_case(materialized.case_path)
    expected = json.loads(materialized.expected_path.read_text(encoding="utf-8"))

    assert case.spec.name == case_id
    assert expected["case_id"] == case_id
    assert case.spec.physics.models["charge"] == {"model": "fixed"}
    assert _all_numbers_are_finite(expected)


@pytest.mark.parametrize("case_id", MICROCASE_IDS)
def test_microcase_generation_is_deterministic(tmp_path: Path, case_id: str) -> None:
    first = materialize_microcase(case_id, tmp_path / "first" / case_id)
    second = materialize_microcase(case_id, tmp_path / "second" / case_id)

    first_case = load_case(first.case_path)
    second_case = load_case(second.case_path)

    assert first_case.content_hash == second_case.content_hash
    assert first.expected_path.read_bytes() == second.expected_path.read_bytes()


def test_public_loader_preserves_continuous_charge_mapping_for_catalog_resolution(
    tmp_path: Path,
) -> None:
    materialized = materialize_microcase("C01", tmp_path / "continuous-charge")
    document = yaml.safe_load(materialized.case_path.read_text(encoding="utf-8"))
    document["physics"]["charge"] = {
        "model": "plasma_continuous",
        "revision": "oml_stationary_maxwellian_debye_huckel_v1",
        "electron_number_density_field": "electron_number_density",
        "positive_ion_number_density_field": "positive_ion_number_density",
        "electron_temperature_field": "electron_temperature",
        "positive_ion_temperature_field": "positive_ion_temperature",
        "positive_ion_velocity_field": "positive_ion_velocity",
        "positive_ion_mass_kg": 6.6335209e-26,
        "applicability": "error",
    }
    materialized.case_path.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )

    case = load_case(materialized.case_path)
    plan = resolve_physics_plan(case.spec.physics.models, case.data.coordinate_system)

    assert plan.evolves_continuous_state
    assert plan.requires_stage_evaluation
    assert plan.resolved_models()["charge"] == {
        "model": "plasma_continuous",
        "revision": "oml_stationary_maxwellian_debye_huckel_v1",
    }


def test_closed_form_and_event_anchors_do_not_depend_on_production_numerics(
    tmp_path: Path,
) -> None:
    expected = {
        case_id: _read_expected(materialize_microcase(case_id, tmp_path / case_id).expected_path)
        for case_id in MICROCASE_IDS
    }

    assert expected["C01"]["position_m"][3][0] == pytest.approx([0.265, -0.345])
    assert expected["C02"]["position_m"][-1][0] == pytest.approx(
        [0.045865886705354923, 0.32293294335267746]
    )
    assert expected["C03"]["velocity_m_s"][-1][0] == pytest.approx(
        [0.029872241020718366, 0.16514905214249524]
    )
    assert expected["C03"]["method_acceptance"]["exponential_midpoint"][
        "comparison_time_s"
    ] == pytest.approx(0.75)
    assert (
        expected["C03"]["method_acceptance"]["exponential_midpoint"]["error_norm"]
        == "component_linf"
    )
    assert expected["C04"]["derived"]["acceleration_m_s2"] == [[-2.0, 1.0], [-1.0, 0.5]]
    assert expected["C05"]["derived"]["acceleration_m_s2"] == [
        [0.0, -10.0],
        [0.0, -7.0],
        [0.0, -8.5],
    ]
    assert expected["C06"]["field_probes"][2]["value"] == pytest.approx([1.76, -2.84])
    assert expected["C07"]["boundary_events"][0]["time_s"] == 1.5
    assert expected["C08"]["zero_time_departure_event_count"] == 0
    assert [event["time_s"] for event in expected["C09"]["boundary_events"]] == [
        0.00048828125,
        0.00146484375,
        0.00244140625,
        0.00341796875,
        0.00439453125,
    ]
    assert expected["C10"]["boundary_events"][0]["candidate_facet_id"] == [1, 2]
    assert expected["C10"]["boundary_events"][0]["velocity_post_m_s"] == pytest.approx(
        [1.0 / math.sqrt(10.0), -3.0 / math.sqrt(10.0)]
    )
    assert expected["C10"]["boundary_events"][1]["law_id"] == "stick"
    assert expected["C10"]["boundary_events"][1]["primary_facet_id"] == 4
    assert expected["C09"]["failure_probes"][0]["failure_reason"] == "numerical_event_budget"
    assert expected["C07"]["acceptance"]["charge_number_exact"]
    assert expected["C07"]["acceptance"]["law_id_and_outcome_exact"]


def test_ballistic_and_epstein_oracles_are_bound_to_canonical_inputs(tmp_path: Path) -> None:
    c01_paths = materialize_microcase("C01", tmp_path / "C01")
    c01 = load_case(c01_paths.case_path)
    c01_expected = _read_expected(c01_paths.expected_path)
    c01_source = c01.data.sources[0]
    time = float(c01_expected["time_s"][3])
    expected_position = c01_source.position_m[0] + time * c01_source.velocity_m_s[0]
    np.testing.assert_allclose(
        c01_expected["position_m"][3][0], expected_position, rtol=0.0, atol=0.0
    )

    for case_id, expected_rate in (("C02", 2.0), ("C03", 4.0)):
        paths = materialize_microcase(case_id, tmp_path / case_id)
        case = load_case(paths.case_path)
        expected = _read_expected(paths.expected_path)
        fields = {field.name: field for field in case.data.fields}
        source = case.data.sources[0]
        drag = case.spec.physics.models["drag"]
        temperature = float(fields["gas_temperature"].values[0, 0])
        density = float(fields["gas_density"].values[0, 0])
        molecular_mass = float(drag["gas_molecular_mass_kg"])
        mean_thermal_speed = math.sqrt(
            8.0 * 1.380649e-23 * temperature / (math.pi * molecular_mass)
        )
        radius = 0.5 * float(source.drag_diameter_m[0])
        rate = (
            (4.0 * math.pi / 3.0)
            * radius**2
            * density
            * mean_thermal_speed
            * float(drag["delta"])
            / float(source.mass_kg[0])
        )

        assert rate == pytest.approx(expected_rate, rel=2.0e-15)
        assert expected["derived"]["rate_s_inverse"] == expected_rate
        assert expected["derived"]["tau_s"] == pytest.approx(1.0 / rate)


def test_force_oracles_are_bound_to_source_authorities(tmp_path: Path) -> None:
    c04_paths = materialize_microcase("C04", tmp_path / "C04")
    c04 = load_case(c04_paths.case_path)
    c04_expected = _read_expected(c04_paths.expected_path)
    electric_field = next(field for field in c04.data.fields if field.name == "electric_field")
    c04_source = c04.data.sources[0]
    accelerations = (
        c04_source.charge_number[:, None]
        * 1.602176634e-19
        * electric_field.values[0]
        / c04_source.mass_kg[:, None]
    )
    np.testing.assert_allclose(
        c04_expected["derived"]["acceleration_m_s2"], accelerations, rtol=0.0, atol=0.0
    )

    c05_paths = materialize_microcase("C05", tmp_path / "C05")
    c05 = load_case(c05_paths.case_path)
    c05_expected = _read_expected(c05_paths.expected_path)
    gas_density = next(field for field in c05.data.fields if field.name == "gas_density")
    c05_source = c05.data.sources[0]
    gravity = np.asarray(
        c05.spec.physics.models["gravity_buoyancy"]["gravity_m_s2"], dtype=np.float64
    )
    density = float(gas_density.values[0, 0])
    accelerations = (1.0 - density * c05_source.displaced_volume_m3 / c05_source.mass_kg)[
        :, None
    ] * gravity
    np.testing.assert_allclose(
        c05_expected["derived"]["acceleration_m_s2"], accelerations, rtol=0.0, atol=0.0
    )


def test_c06_oracle_binds_two_cell_p1_and_mapped_q1_inputs(tmp_path: Path) -> None:
    paths = materialize_microcase("C06", tmp_path / "C06")
    case = load_case(paths.case_path)
    expected = _read_expected(paths.expected_path)
    layouts = {layout.name: layout for layout in case.data.layouts}
    fields = {field.name: field for field in case.data.fields}

    np.testing.assert_array_equal(layouts["p1"].connectivity, [[0, 1, 2], [1, 3, 2]])
    np.testing.assert_allclose(
        fields["p1_affine"].values,
        [[1.0, -2.0], [5.0, -1.0], [-2.0, 2.0], [2.0, 3.0]],
        rtol=0.0,
        atol=0.0,
    )
    shared_probe = expected["field_probes"][1]
    first_cell_value = np.asarray([0.0, 0.5, 0.5]) @ fields["p1_affine"].values[[0, 1, 2]]
    second_cell_value = np.asarray([0.5, 0.0, 0.5]) @ fields["p1_affine"].values[[1, 3, 2]]
    np.testing.assert_allclose(
        shared_probe["value_from_each_cell"],
        [first_cell_value, second_cell_value],
        rtol=0.0,
        atol=0.0,
    )

    reference = np.asarray([0.2, -0.4])
    xi, eta = reference
    weights = 0.25 * np.asarray(
        [
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ]
    )
    q1_probe = expected["field_probes"][2]
    np.testing.assert_allclose(q1_probe["shape_weights"], weights, rtol=0.0, atol=1.0e-16)
    np.testing.assert_allclose(
        q1_probe["position_m"], weights @ layouts["q1"].nodes_m, rtol=0.0, atol=1.0e-16
    )
    np.testing.assert_allclose(
        q1_probe["value"], weights @ fields["q1_bilinear"].values, rtol=0.0, atol=1.0e-15
    )


def test_boundary_oracles_are_bound_to_geometry_laws_and_identity(tmp_path: Path) -> None:
    c08_paths = materialize_microcase("C08", tmp_path / "C08")
    c08 = load_case(c08_paths.case_path)
    c08_expected = _read_expected(c08_paths.expected_path)
    c08_source = c08.spec.sources[0]
    assert c08_source.parameters["particle_id_start"] == 801
    assert c08_expected["boundary_events"][0]["particle_id"] == 801
    assert [boundary.boundary_group for boundary in c08.spec.boundaries] == [
        "collector",
        "mirror",
        "caps",
    ]

    c09_paths = materialize_microcase("C09", tmp_path / "C09")
    c09 = load_case(c09_paths.case_path)
    c09_expected = _read_expected(c09_paths.expected_path)
    gap = float(c09.data.geometry.nodes_m[1, 0] - c09.data.geometry.nodes_m[0, 0])
    derived_hit_times = [gap / 2.0 + index * gap for index in range(5)]
    assert [event["time_s"] for event in c09_expected["boundary_events"]] == derived_hit_times
    success_probe = c09_expected["success_probes"][0]
    assert success_probe["spec_override"] == {
        "solver.event.max_interactions_per_step": 1,
        "solver.event.max_refinements": 2,
    }
    assert success_probe["expected_run_status"] == "complete"
    assert success_probe["expected_boundary_events"] == c09_expected["boundary_events"]
    assert success_probe["expected_final"] == c09_expected["final"]
    failure_probe = c09_expected["failure_probes"][0]
    assert failure_probe["spec_override"] == {
        "solver.event.max_interactions_per_step": 1,
        "solver.event.max_refinements": 1,
    }
    assert (
        failure_probe["expected_processed_boundary_events"] == c09_expected["boundary_events"][:2]
    )
    assert failure_probe["expected_failure_time_s"] == derived_hit_times[1]
    assert failure_probe["next_unprocessed_boundary_event_ordinal"] == 2
    assert failure_probe["next_unprocessed_boundary_event_time_s"] == derived_hit_times[2]

    c10_paths = materialize_microcase("C10", tmp_path / "C10")
    c10 = load_case(c10_paths.case_path)
    c10_expected = _read_expected(c10_paths.expected_path)
    laws = {boundary.boundary_group: boundary for boundary in c10.spec.boundaries}
    assert (laws["priority_collector"].priority, laws["priority_collector"].law) == (5, "stick")
    assert (laws["priority_mirror"].priority, laws["priority_mirror"].law) == (20, "specular")
    np.testing.assert_array_equal(c10.data.geometry.boundary.group_id[[4, 5]], [2, 3])
    priority_event = c10_expected["boundary_events"][1]
    assert priority_event["candidate_facet_id"] == [4, 5]
    assert priority_event["primary_facet_id"] == 4
    assert priority_event["outcome"] == "stuck"
    policy_probe = c10_expected["policy_failure_probes"][1]
    assert policy_probe["selected_primary_facet_id"] == 10
    assert policy_probe["selected_velocity_post_m_s"] == [-1.0, 1.0]
    ignored_normal = np.asarray(policy_probe["candidates"][1]["normal"])
    selected_velocity = np.asarray(policy_probe["selected_velocity_post_m_s"])
    assert float(np.dot(selected_velocity, ignored_normal)) > 0.0
    assert policy_probe["failure_reason"] == "indeterminate_boundary_policy"


def _read_expected(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _all_numbers_are_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_all_numbers_are_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_all_numbers_are_finite(item) for item in value.values())
    return False
