from __future__ import annotations

import json
import math
import os
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import chamber_particles.engine as engine_module
import chamber_particles.output as output_module
from chamber_particles import SimulationError, load_case, open_result, simulate
from chamber_particles.case_format import (
    BoundaryData,
    FieldData,
    GeometryData,
    P1TriLayout,
    Q1QuadLayout,
    RegularLayout,
    write,
)
from chamber_particles.physics.forces import (
    BOLTZMANN_J_K,
    ELEMENTARY_CHARGE_C,
    VACUUM_PERMITTIVITY_F_M,
)
from tests.verification.microcases import materialize_microcase

_CONTINUOUS_CHARGE_ION_MASS_KG = 6.6335209e-26
_CONTINUOUS_CHARGE_ELECTRON_TEMPERATURE_K = 1.0e5
_CONTINUOUS_CHARGE_ION_TEMPERATURE_K = 1.0e4
_AGGREGATE_CHARGE_REVISION = "aggregate_relative_drift_regularized_two_current_v1"
_SIGNED_AGGREGATE_CHARGE_REVISION = "aggregate_relative_drift_regularized_three_current_v1"
_BARNES_ION_DRAG_REVISION = (
    "barnes_collisionless_effective_speed_single_positive_ion_negative_debye_huckel_v1"
)


@pytest.mark.parametrize(
    ("case_id", "position_atol_m", "velocity_atol_m_s", "path_kind"),
    [
        ("C02", 5.0e-5, 8.0e-5, "rk4_dense"),
        ("C03", 6.0e-5, 2.0e-4, "rk4_dense"),
        ("C04", 2.0e-13, 2.0e-13, "quadratic_exact"),
        ("C05", 2.0e-13, 2.0e-13, "quadratic_exact"),
    ],
)
def test_force_coupled_microcases_match_analytic_time_series(
    tmp_path: Path,
    case_id: str,
    position_atol_m: float,
    velocity_atol_m_s: float,
    path_kind: str,
) -> None:
    paths = materialize_microcase(case_id, tmp_path / case_id)
    expected = json.loads(paths.expected_path.read_text(encoding="utf-8"))
    case = load_case(paths.case_path)
    output = tmp_path / f"result-{case_id}"

    simulate(case, output)
    result = open_result(output)
    frames = list(result.iter_frames())

    assert [frame.time_s for frame in frames] == expected["time_s"]
    for index, frame in enumerate(frames):
        np.testing.assert_allclose(
            frame.position_m,
            expected["position_m"][index],
            rtol=0.0,
            atol=position_atol_m,
        )
        np.testing.assert_allclose(
            frame.velocity_m_s,
            expected["velocity_m_s"][index],
            rtol=0.0,
            atol=velocity_atol_m_s,
        )
        np.testing.assert_array_equal(frame.charge_number, expected["charge_number"][index])

    final = result.read_final()
    np.testing.assert_allclose(
        final.position_m,
        expected["position_m"][-1],
        rtol=0.0,
        atol=position_atol_m,
    )
    np.testing.assert_allclose(
        final.velocity_m_s,
        expected["velocity_m_s"][-1],
        rtol=0.0,
        atol=velocity_atol_m_s,
    )
    np.testing.assert_array_equal(final.charge_number, expected["charge_number"][-1])
    assert result.read_boundary_events().particle_id.size == 0
    assert result.manifest["field_location_revision"] == "field_location_v4"
    assert result.manifest["required_field_revision"] == "required_field_rz_axis_domain_regular_v3"
    expected_enclosure_revision = (
        "rk4_global_abs_enclosure_v2" if path_kind == "rk4_dense" else None
    )
    assert result.manifest["rk4_enclosure_revision"] == expected_enclosure_revision
    expected_dense_revision = (
        "rk4_position_hermite_state_extension_v3" if path_kind == "rk4_dense" else None
    )
    assert result.manifest["rk4_dense_path_revision"] == expected_dense_revision
    assert result.manifest["resolved"]["path_kind"] == path_kind
    assert result.manifest["resolved"]["physics_models"] == {
        category: {
            "model": values["model"],
            "revision": {
                "charge": "fixed_charge_v1",
                "drag": "epstein_linear_v1",
                "electric": "electric_coulomb_v1",
                "gravity_buoyancy": "gravity_buoyancy_standard_v1",
            }[category],
        }
        for category, values in case.spec.physics.models.items()
    }
    assert result.manifest["resolved"]["required_fields"]


def test_boundaryless_quadratic_result_is_independent_of_output_schedule(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C05", tmp_path / "C05-output-independence")
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["output"]["trajectories"] = None
    no_frames_path = paths.case_path.with_name("case-no-frames.yaml")
    no_frames_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    sparse_document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    sparse_document["output"]["trajectories"]["schedule"] = {
        "explicit_times_s": [0.37 * float(sparse_document["time"]["end_s"])]
    }
    sparse_path = paths.case_path.with_name("case-sparse-frame.yaml")
    sparse_path.write_text(yaml.safe_dump(sparse_document, sort_keys=False), encoding="utf-8")

    simulate(load_case(paths.case_path), tmp_path / "with-dense-frames")
    simulate(load_case(sparse_path), tmp_path / "with-sparse-frame")
    simulate(load_case(no_frames_path), tmp_path / "without-frames")
    with_dense_frames = open_result(tmp_path / "with-dense-frames").read_final()
    with_sparse_frame = open_result(tmp_path / "with-sparse-frame").read_final()
    without_frames = open_result(tmp_path / "without-frames").read_final()

    for result in (with_sparse_frame, without_frames):
        np.testing.assert_array_equal(with_dense_frames.position_m, result.position_m)
        np.testing.assert_array_equal(with_dense_frames.velocity_m_s, result.velocity_m_s)
        np.testing.assert_array_equal(with_dense_frames.charge_number, result.charge_number)


def _continuous_charge_case(
    directory: Path,
    *,
    integrator: str = "rk4_fixed",
    axisymmetric_rz: bool = False,
    number_density_m3: float = 1.0e6,
    dt_s: float = 0.25,
    electric: bool = True,
    output: bool = True,
    charge_revision: str = "oml_stationary_maxwellian_debye_huckel_v1",
    ion_velocity_m_s: tuple[float, float] = (0.0, 0.0),
    maximum_ion_drift_ratio: float | None = None,
    maximum_relative_ion_speed_m_s: float = 1.0e4,
    negative_ion_density_m3: float = 0.0,
    negative_ion_velocity_m_s: tuple[float, float] = (0.0, 0.0),
    far_electric_field_V_m: tuple[float, float] | None = None,
    far_ion_velocity_m_s: tuple[float, float] | None = None,
) -> Path:
    paths = materialize_microcase("C07", directory)
    case = load_case(paths.case_path)
    radial_shift = np.asarray([1.5, 0.0]) if axisymmetric_rz else np.zeros(2)
    original = case.data.sources[0]
    source = replace(
        original,
        position_m=original.position_m + radial_shift,
        charge_number=np.zeros_like(original.charge_number),
    )
    coordinate_system = "axisymmetric_rz" if axisymmetric_rz else "cartesian_xy"
    vector_components = ("r", "z") if axisymmetric_rz else ("x", "y")
    vector_basis = "axisymmetric_rz" if axisymmetric_rz else "cartesian_xy"
    layout_x = np.asarray(
        [radial_shift[0], radial_shift[0] + 1.0]
        if far_electric_field_V_m is None and far_ion_velocity_m_s is None
        else [radial_shift[0], radial_shift[0] + 1.0, radial_shift[0] + 2.0],
        dtype="<f8",
    )
    layout = RegularLayout(
        "plasma",
        layout_x,
        np.asarray([0.0, 1.0], dtype="<f8"),
        np.ones((layout_x.size - 1, 1), dtype="<u1"),
    )

    def uniform_field(
        name: str,
        value: tuple[float, ...],
        components: tuple[str, ...],
        basis: str,
        unit: str,
    ) -> FieldData:
        values = np.repeat(
            np.asarray([value], dtype="<f8"),
            layout.axis0_m.size * layout.axis1_m.size,
            axis=0,
        )
        if name == "electric_field" and far_electric_field_V_m is not None:
            values.reshape(layout.axis0_m.size, layout.axis1_m.size, -1)[-1, :, :] = (
                far_electric_field_V_m
            )
        if name == "ion_velocity" and far_ion_velocity_m_s is not None:
            values.reshape(layout.axis0_m.size, layout.axis1_m.size, -1)[-1, :, :] = (
                far_ion_velocity_m_s
            )
        return FieldData(name, "plasma", "node", components, basis, values, unit)

    fields = [
        uniform_field(
            "electron_density",
            (number_density_m3,),
            ("value",),
            "scalar",
            "1/m^3",
        ),
        uniform_field(
            "ion_density",
            (number_density_m3,),
            ("value",),
            "scalar",
            "1/m^3",
        ),
        uniform_field(
            "ion_velocity",
            ion_velocity_m_s,
            vector_components,
            vector_basis,
            "m/s",
        ),
    ]
    if charge_revision in (_AGGREGATE_CHARGE_REVISION, _SIGNED_AGGREGATE_CHARGE_REVISION):
        fields.extend(
            (
                uniform_field(
                    "electron_voltage",
                    (
                        BOLTZMANN_J_K
                        * _CONTINUOUS_CHARGE_ELECTRON_TEMPERATURE_K
                        / ELEMENTARY_CHARGE_C,
                    ),
                    ("value",),
                    "scalar",
                    "V",
                ),
                uniform_field(
                    "ion_voltage",
                    (BOLTZMANN_J_K * _CONTINUOUS_CHARGE_ION_TEMPERATURE_K / ELEMENTARY_CHARGE_C,),
                    ("value",),
                    "scalar",
                    "V",
                ),
                uniform_field(
                    "ion_mass",
                    (_CONTINUOUS_CHARGE_ION_MASS_KG,),
                    ("value",),
                    "scalar",
                    "kg",
                ),
                uniform_field(
                    "screening_length",
                    (1.0e-3,),
                    ("value",),
                    "scalar",
                    "m",
                ),
            )
        )
        if charge_revision == _SIGNED_AGGREGATE_CHARGE_REVISION:
            fields.extend(
                (
                    uniform_field(
                        "negative_ion_density",
                        (negative_ion_density_m3,),
                        ("value",),
                        "scalar",
                        "1/m^3",
                    ),
                    uniform_field(
                        "negative_ion_voltage",
                        (
                            BOLTZMANN_J_K
                            * _CONTINUOUS_CHARGE_ION_TEMPERATURE_K
                            / ELEMENTARY_CHARGE_C,
                        ),
                        ("value",),
                        "scalar",
                        "V",
                    ),
                    uniform_field(
                        "negative_ion_velocity",
                        negative_ion_velocity_m_s,
                        vector_components,
                        vector_basis,
                        "m/s",
                    ),
                    uniform_field(
                        "negative_ion_mass",
                        (_CONTINUOUS_CHARGE_ION_MASS_KG,),
                        ("value",),
                        "scalar",
                        "kg",
                    ),
                )
            )
    else:
        fields.extend(
            (
                uniform_field(
                    "electron_temperature",
                    (_CONTINUOUS_CHARGE_ELECTRON_TEMPERATURE_K,),
                    ("value",),
                    "scalar",
                    "K",
                ),
                uniform_field(
                    "ion_temperature",
                    (_CONTINUOUS_CHARGE_ION_TEMPERATURE_K,),
                    ("value",),
                    "scalar",
                    "K",
                ),
            )
        )
    if electric:
        fields.append(
            uniform_field(
                "electric_field",
                (1.0, -0.5),
                vector_components,
                vector_basis,
                "V/m",
            )
        )
    geometry = replace(case.data.geometry, nodes_m=case.data.geometry.nodes_m + radial_shift)
    data_path = paths.case_path.with_name("continuous-charge.h5")
    info = write(
        data_path,
        replace(
            case.data,
            coordinate_system=coordinate_system,
            geometry=geometry,
            layouts=(layout,),
            fields=tuple(fields),
            sources=(source,),
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["motion"]["mode"] = "axisymmetric_rz_meridional" if axisymmetric_rz else "cartesian_xy"
    document["time"]["dt_s"] = dt_s
    document["solver"]["integrator"] = integrator
    if charge_revision in (_AGGREGATE_CHARGE_REVISION, _SIGNED_AGGREGATE_CHARGE_REVISION):
        document["physics"]["charge"] = {
            "model": "plasma_continuous",
            "revision": charge_revision,
            "electron_number_density_field": "electron_density",
            "positive_ion_number_density_field": "ion_density",
            "electron_thermal_voltage_field": "electron_voltage",
            "positive_ion_thermal_voltage_field": "ion_voltage",
            "positive_ion_velocity_field": "ion_velocity",
            "effective_positive_ion_mass_field": "ion_mass",
            "screening_length_field": "screening_length",
            "maximum_relative_ion_speed_m_s": maximum_relative_ion_speed_m_s,
            "applicability": "error",
        }
        if charge_revision == _SIGNED_AGGREGATE_CHARGE_REVISION:
            document["physics"]["charge"].update(
                {
                    "negative_ion_number_density_field": "negative_ion_density",
                    "negative_ion_thermal_voltage_field": "negative_ion_voltage",
                    "negative_ion_velocity_field": "negative_ion_velocity",
                    "effective_negative_ion_mass_field": "negative_ion_mass",
                }
            )
    else:
        document["physics"]["charge"] = {
            "model": "plasma_continuous",
            "revision": charge_revision,
            "electron_number_density_field": "electron_density",
            "positive_ion_number_density_field": "ion_density",
            "electron_temperature_field": "electron_temperature",
            "positive_ion_temperature_field": "ion_temperature",
            "positive_ion_velocity_field": "ion_velocity",
            "positive_ion_mass_kg": _CONTINUOUS_CHARGE_ION_MASS_KG,
            "applicability": "error",
        }
        if maximum_ion_drift_ratio is not None:
            document["physics"]["charge"]["maximum_ion_drift_ratio"] = maximum_ion_drift_ratio
    if electric:
        document["physics"]["electric"] = {
            "model": "coulomb",
            "revision": "electric_coulomb_v1",
            "electric_field": "electric_field",
        }
    document["output"] = (
        {
            "trajectories": {
                "selection": "all",
                "schedule": {"explicit_times_s": [0.0, 0.75, 1.5, 2.0]},
            },
            "probes": {
                "particle_ids": [701],
                "schedule": {"explicit_times_s": [0.75, 1.5]},
            },
        }
        if output
        else {"trajectories": None}
    )
    case_path = paths.case_path.with_name("continuous-charge.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


@pytest.mark.parametrize(
    ("integrator", "path_kind", "charge_revision", "ion_velocity_m_s", "maximum_drift"),
    [
        (
            "rk4_fixed",
            "rk4_dense",
            "oml_stationary_maxwellian_debye_huckel_v1",
            (0.0, 0.0),
            None,
        ),
        (
            "exponential_midpoint",
            "exponential_midpoint_reintegrated",
            "oml_stationary_maxwellian_debye_huckel_v1",
            (0.0, 0.0),
            None,
        ),
        (
            "rk4_fixed",
            "rk4_dense",
            "oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1",
            (500.0, 0.0),
            1.0,
        ),
        (
            "exponential_midpoint",
            "exponential_midpoint_reintegrated",
            "oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1",
            (500.0, 0.0),
            1.0,
        ),
    ],
)
def test_continuous_charge_reaches_wall_through_shared_event_path(
    tmp_path: Path,
    integrator: str,
    path_kind: str,
    charge_revision: str,
    ion_velocity_m_s: tuple[float, float],
    maximum_drift: float | None,
) -> None:
    case_path = _continuous_charge_case(
        tmp_path / f"continuous-charge-{integrator}",
        integrator=integrator,
        charge_revision=charge_revision,
        ion_velocity_m_s=ion_velocity_m_s,
        maximum_ion_drift_ratio=maximum_drift,
    )
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    unframed_document = json.loads(json.dumps(document))
    unframed_document["output"] = {"trajectories": None}
    unframed_path = case_path.with_name("continuous-charge-unframed.yaml")
    unframed_path.write_text(
        yaml.safe_dump(unframed_document, sort_keys=False),
        encoding="utf-8",
    )

    output = tmp_path / f"continuous-charge-result-{integrator}"
    unframed_output = tmp_path / f"continuous-charge-unframed-result-{integrator}"
    simulate(load_case(case_path), output)
    simulate(load_case(unframed_path), unframed_output)
    result = open_result(output)
    unframed_result = open_result(unframed_output)
    final = result.read_final()
    events = result.read_boundary_events()
    unframed_final = unframed_result.read_final()
    unframed_events = unframed_result.read_boundary_events()
    frames = list(result.iter_frames())
    probes = list(result.iter_probes())

    np.testing.assert_array_equal(final.lifecycle, [2])
    assert events.particle_id.tolist() == [701]
    assert float(events.charge_number_pre[0]) < 0.0
    assert float(events.velocity_pre_m_s[0, 0]) < 0.5
    np.testing.assert_array_equal(events.charge_number_post, events.charge_number_pre)
    np.testing.assert_array_equal(final.charge_number, events.charge_number_post)
    assert [frame.time_s for frame in frames] == [0.0, 0.75, 1.5, 2.0]
    assert [probe.time_s for probe in probes] == [0.75, 1.5]
    assert float(frames[1].charge_number[0]) < 0.0
    np.testing.assert_array_equal(frames[-1].charge_number, final.charge_number)
    assert float(events.charge_number_post[0]) < float(probes[-1].charge_number[0])
    for name in ("position_m", "velocity_m_s", "charge_number", "lifecycle"):
        np.testing.assert_array_equal(getattr(final, name), getattr(unframed_final, name))
    for name in (
        "time_s",
        "particle_id",
        "position_m",
        "velocity_pre_m_s",
        "velocity_post_m_s",
        "charge_number_pre",
        "charge_number_post",
    ):
        np.testing.assert_array_equal(getattr(events, name), getattr(unframed_events, name))
    assert result.manifest["resolved"]["path_kind"] == path_kind
    assert result.manifest["resolved"]["physics_models"]["charge"] == {
        "model": "plasma_continuous",
        "revision": charge_revision,
    }
    assert result.manifest["resolved"]["physics_models"]["electric"] == {
        "model": "coulomb",
        "revision": "electric_coulomb_v1",
    }
    assert result.manifest["maximum_dt_charge_lipschitz"] <= 0.5
    assert result.manifest["failure_reason_counts"]["integrator_accuracy"] == 0


@pytest.mark.parametrize(
    ("integrator", "axisymmetric_rz", "path_kind"),
    [
        ("rk4_fixed", False, "rk4_dense"),
        ("exponential_midpoint", True, "exponential_midpoint_reintegrated"),
    ],
)
def test_aggregate_charge_updates_public_state_in_xy_and_rz(
    tmp_path: Path,
    integrator: str,
    axisymmetric_rz: bool,
    path_kind: str,
) -> None:
    case_path = _continuous_charge_case(
        tmp_path / f"aggregate-charge-{integrator}",
        integrator=integrator,
        axisymmetric_rz=axisymmetric_rz,
        charge_revision=_AGGREGATE_CHARGE_REVISION,
        ion_velocity_m_s=(80.0, 20.0),
    )
    output = tmp_path / f"aggregate-charge-result-{integrator}"

    simulate(load_case(case_path), output)
    result = open_result(output)
    final = result.read_final()
    frames = list(result.iter_frames())
    events = result.read_boundary_events()

    assert float(frames[1].charge_number[0]) < 0.0
    np.testing.assert_array_equal(final.charge_number, events.charge_number_post)
    assert result.manifest["resolved"]["path_kind"] == path_kind
    assert result.manifest["resolved"]["physics_models"]["charge"] == {
        "model": "plasma_continuous",
        "revision": _AGGREGATE_CHARGE_REVISION,
    }
    assert result.manifest["data_coordinate_system"] == (
        "axisymmetric_rz" if axisymmetric_rz else "cartesian_xy"
    )
    assert result.manifest["failure_reason_counts"]["integrator_accuracy"] == 0


def test_signed_aggregate_charge_public_api_preserves_zero_density_limit(
    tmp_path: Path,
) -> None:
    configurations = (
        ("two-current", _AGGREGATE_CHARGE_REVISION, 0.0),
        ("three-current-zero", _SIGNED_AGGREGATE_CHARGE_REVISION, 0.0),
        ("three-current-finite", _SIGNED_AGGREGATE_CHARGE_REVISION, 1.0e9),
    )
    results = {}
    for name, revision, negative_density in configurations:
        case_path = _continuous_charge_case(
            tmp_path / name,
            dt_s=0.01,
            output=False,
            charge_revision=revision,
            ion_velocity_m_s=(80.0, 20.0),
            negative_ion_density_m3=negative_density,
            negative_ion_velocity_m_s=(-35.0, 15.0),
        )
        output = tmp_path / f"{name}-result"
        simulate(load_case(case_path), output)
        results[name] = open_result(output)

    two_current = results["two-current"].read_final()
    zero_negative = results["three-current-zero"].read_final()
    finite_negative = results["three-current-finite"].read_final()
    for field in ("position_m", "velocity_m_s", "charge_number", "lifecycle"):
        np.testing.assert_array_equal(getattr(two_current, field), getattr(zero_negative, field))
    assert float(finite_negative.charge_number[0]) < float(zero_negative.charge_number[0])
    assert results["three-current-zero"].manifest["resolved"]["physics_models"]["charge"] == {
        "model": "plasma_continuous",
        "revision": _SIGNED_AGGREGATE_CHARGE_REVISION,
    }


def test_exponential_midpoint_uses_local_enclosure_for_unused_extreme_cell(
    tmp_path: Path,
) -> None:
    case_path = _continuous_charge_case(
        tmp_path / "local-force-enclosure",
        integrator="exponential_midpoint",
        dt_s=0.25,
        output=False,
        charge_revision=_SIGNED_AGGREGATE_CHARGE_REVISION,
        ion_velocity_m_s=(80.0, 20.0),
        negative_ion_density_m3=1.0e9,
        negative_ion_velocity_m_s=(-35.0, 15.0),
        far_electric_field_V_m=(1.0e18, -1.0e18),
        far_ion_velocity_m_s=(1.0e8, -1.0e8),
    )
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["time"]["end_s"] = 0.25
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / "local-force-enclosure-result"

    summary = simulate(load_case(case_path), output)
    result = open_result(output)
    final = result.read_final()

    assert summary.failure_event_count == 0
    assert result.manifest["event_refinement"]["maximum_refinement_depth"] <= 1
    assert np.isfinite(final.position_m).all()
    assert np.isfinite(final.velocity_m_s).all()
    assert np.isfinite(final.charge_number).all()
    memory_plan = result.manifest["memory_plan"]
    assert memory_plan["revision"] == "solver_owned_memory_plan_v14"
    assert memory_plan["certificate_work_bytes_per_particle"] == 544
    assert memory_plan["components"]["slab_certificate_work"] == (
        544 * memory_plan["slab_particles"]
    )


@pytest.mark.parametrize(
    ("integrator", "minimum_order"),
    [("rk4_fixed", 3.5), ("exponential_midpoint", 1.8)],
)
def test_aggregate_charge_coupled_time_refinement(
    tmp_path: Path,
    integrator: str,
    minimum_order: float,
) -> None:
    states: list[np.ndarray] = []
    for index, dt_s in enumerate((0.015625, 0.0078125, 0.00390625, 0.001953125)):
        case_path = _continuous_charge_case(
            tmp_path / f"aggregate-charge-refinement-{integrator}-{index}",
            integrator=integrator,
            dt_s=dt_s,
            number_density_m3=1.0e9,
            output=False,
            charge_revision=_AGGREGATE_CHARGE_REVISION,
            ion_velocity_m_s=(80.0, 20.0),
        )
        document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
        document["time"]["end_s"] = 0.5
        case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        output = tmp_path / f"aggregate-charge-refinement-result-{integrator}-{index}"

        simulate(load_case(case_path), output)
        result = open_result(output)
        assert result.manifest["event_refinement"]["refinements"] == 0
        final = result.read_final()
        states.append(
            np.concatenate((final.position_m[0], final.velocity_m_s[0], final.charge_number[:1]))
        )

    differences = np.stack([np.abs(states[index] - states[index + 1]) for index in range(3)])
    assert bool((differences > 0.0).all())
    component_orders = np.log2(differences[:-1] / differences[1:])
    assert float(np.min(component_orders)) >= minimum_order


def test_continuous_charge_stability_gate_is_owned_by_explicit_rk4(
    tmp_path: Path,
) -> None:
    number_density_m3 = 1.0e9
    probe_dt_s = 0.25
    probe_case = _continuous_charge_case(
        tmp_path / "continuous-charge-hl-probe",
        number_density_m3=number_density_m3,
        dt_s=probe_dt_s,
        electric=False,
        output=False,
    )
    probe_output = tmp_path / "continuous-charge-hl-probe-result"
    simulate(load_case(probe_case), probe_output)
    charge_lipschitz = (
        open_result(probe_output).manifest["maximum_dt_charge_lipschitz"] / probe_dt_s
    )
    limit_dt_s = 0.5 / charge_lipschitz
    accepted_case = _continuous_charge_case(
        tmp_path / "continuous-charge-hl-limit",
        integrator="rk4_fixed",
        number_density_m3=number_density_m3,
        dt_s=limit_dt_s,
        electric=False,
        output=False,
    )
    accepted_output = tmp_path / "continuous-charge-hl-limit-result"

    simulate(load_case(accepted_case), accepted_output)

    assert open_result(accepted_output).manifest["maximum_dt_charge_lipschitz"] == 0.5
    rejected_case = _continuous_charge_case(
        tmp_path / "continuous-charge-hl-excess",
        integrator="rk4_fixed",
        number_density_m3=number_density_m3,
        dt_s=limit_dt_s * (1.0 + 1.0e-12),
        electric=False,
        output=False,
    )
    with pytest.raises(
        SimulationError,
        match=r"continuous charge requires dt \* charge_lipschitz <= 0.5 for explicit integration",
    ):
        simulate(load_case(rejected_case), tmp_path / "continuous-charge-hl-excess-result")

    stable_case = _continuous_charge_case(
        tmp_path / "continuous-charge-stiff-exponential",
        integrator="exponential_midpoint",
        number_density_m3=number_density_m3,
        dt_s=4.1 * limit_dt_s,
        electric=False,
        output=False,
    )
    stable_output = tmp_path / "continuous-charge-stiff-exponential-result"
    simulate(load_case(stable_case), stable_output)
    stable = open_result(stable_output)

    assert stable.manifest["maximum_dt_charge_lipschitz"] > 2.0
    assert stable.manifest["failure_reason_counts"].get("integrator_numerical", 0) == 0
    assert stable.manifest["failure_reason_counts"].get("integrator_accuracy", 0) == 0


def test_continuous_charge_coulomb_feedback_reuses_xy_and_rz_paths(tmp_path: Path) -> None:
    charge_revision = "oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1"
    xy_case = _continuous_charge_case(
        tmp_path / "continuous-charge-xy",
        charge_revision=charge_revision,
        ion_velocity_m_s=(500.0, 80.0),
        maximum_ion_drift_ratio=1.0,
    )
    rz_case = _continuous_charge_case(
        tmp_path / "continuous-charge-rz",
        axisymmetric_rz=True,
        charge_revision=charge_revision,
        ion_velocity_m_s=(500.0, 80.0),
        maximum_ion_drift_ratio=1.0,
    )
    xy_output = tmp_path / "continuous-charge-xy-result"
    rz_output = tmp_path / "continuous-charge-rz-result"

    simulate(load_case(xy_case), xy_output)
    simulate(load_case(rz_case), rz_output)

    xy = open_result(xy_output)
    rz = open_result(rz_output)
    radial_shift = np.asarray([1.5, 0.0])
    xy_final = xy.read_final()
    rz_final = rz.read_final()
    np.testing.assert_allclose(
        rz_final.position_m - radial_shift,
        xy_final.position_m,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        rz_final.velocity_m_s,
        xy_final.velocity_m_s,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        rz_final.charge_number,
        xy_final.charge_number,
        rtol=2.0e-13,
        atol=2.0e-14,
    )
    xy_events = xy.read_boundary_events()
    rz_events = rz.read_boundary_events()
    np.testing.assert_allclose(rz_events.time_s, xy_events.time_s, rtol=0.0, atol=2.0e-14)
    np.testing.assert_allclose(
        rz_events.position_m - radial_shift,
        xy_events.position_m,
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        rz_events.charge_number_pre,
        xy_events.charge_number_pre,
        rtol=2.0e-13,
        atol=2.0e-14,
    )
    assert rz.manifest["data_coordinate_system"] == "axisymmetric_rz"
    assert rz.manifest["motion_mode"] == "axisymmetric_rz_meridional"


@pytest.mark.parametrize(
    ("label", "charge_revision", "ion_velocity_m_s", "maximum_ion_drift_ratio"),
    [
        (
            "shifted",
            "oml_shifted_maxwellian_single_ion_negative_debye_huckel_v1",
            (500.0, 80.0),
            1.0,
        ),
        (
            "aggregate",
            _AGGREGATE_CHARGE_REVISION,
            (80.0, 20.0),
            None,
        ),
    ],
)
def test_continuous_charge_checkpoint_resume_preserves_public_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    charge_revision: str,
    ion_velocity_m_s: tuple[float, float],
    maximum_ion_drift_ratio: float | None,
) -> None:
    monkeypatch.setattr(engine_module, "_DURABLE_COMMIT_MINIMUM_WORK", 1)
    case_path = _continuous_charge_case(
        tmp_path / f"{label}-charge-resume-case",
        dt_s=2.0 / 130.0,
        charge_revision=charge_revision,
        ion_velocity_m_s=ion_velocity_m_s,
        maximum_ion_drift_ratio=maximum_ion_drift_ratio,
    )
    case = load_case(case_path)
    uninterrupted_path = tmp_path / f"{label}-charge-uninterrupted"
    interrupted_path = tmp_path / f"{label}-charge-interrupted"
    simulate(case, uninterrupted_path)

    real_replace = os.replace
    latest_count = 0

    def replace_and_fail(source: Any, destination: Any) -> None:
        nonlocal latest_count
        destination_path = Path(destination)
        real_replace(source, destination)
        if destination_path.name == "LATEST":
            latest_count += 1
            if latest_count == 2:
                raise OSError("injected failure after continuous-charge checkpoint commit")

    with monkeypatch.context() as patch:
        patch.setattr(output_module.os, "replace", replace_and_fail)
        with pytest.raises(SimulationError):
            simulate(case, interrupted_path)

    recovered = open_result(interrupted_path, recovery=True)
    recovered_steps = recovered.read_lifecycle_series().time_s.size
    assert 0 < recovered_steps < 130
    simulate(case, interrupted_path)
    _assert_public_result_identity(
        open_result(uninterrupted_path),
        open_result(interrupted_path),
    )


@pytest.mark.parametrize("case_id", ["C02", "C03"])
def test_boundaryless_general_rk4_result_is_independent_of_output_schedule(
    tmp_path: Path,
    case_id: str,
) -> None:
    paths = materialize_microcase(case_id, tmp_path / case_id)
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["output"]["trajectories"] = None
    no_frames_path = paths.case_path.with_name("case-no-frames.yaml")
    no_frames_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    sparse_document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    sparse_document["output"]["trajectories"]["schedule"] = {
        "explicit_times_s": [0.37 * float(sparse_document["time"]["end_s"])]
    }
    sparse_path = paths.case_path.with_name("case-sparse-frame.yaml")
    sparse_path.write_text(yaml.safe_dump(sparse_document, sort_keys=False), encoding="utf-8")

    simulate(load_case(paths.case_path), tmp_path / "with-dense-frames")
    simulate(load_case(sparse_path), tmp_path / "with-sparse-frame")
    simulate(load_case(no_frames_path), tmp_path / "without-frames")
    with_dense_frames = open_result(tmp_path / "with-dense-frames").read_final()
    with_sparse_frame = open_result(tmp_path / "with-sparse-frame").read_final()
    without_frames = open_result(tmp_path / "without-frames").read_final()

    for result in (with_sparse_frame, without_frames):
        np.testing.assert_array_equal(with_dense_frames.position_m, result.position_m)
        np.testing.assert_array_equal(with_dense_frames.velocity_m_s, result.velocity_m_s)
        np.testing.assert_array_equal(with_dense_frames.charge_number, result.charge_number)


def test_safe_nonuniform_regular_field_uses_general_rk4_enclosure(tmp_path: Path) -> None:
    """A globally nonuniform field may run when its complete RK4 tube is inside support."""

    paths = materialize_microcase("C04", tmp_path / "nonuniform-safe")
    case = load_case(paths.case_path)
    original = case.data.sources[0]
    source = replace(
        original,
        particle_id=original.particle_id[:1].copy(),
        release_time_s=original.release_time_s[:1].copy(),
        position_m=np.asarray([[0.3, 0.0]], dtype="<f8"),
        velocity_m_s=np.asarray([[-0.1, 0.0]], dtype="<f8"),
        charge_number=original.charge_number[:1].copy(),
        mass_kg=original.mass_kg[:1].copy(),
        drag_diameter_m=original.drag_diameter_m[:1].copy(),
        electrostatic_radius_m=original.electrostatic_radius_m[:1].copy(),
        displaced_volume_m3=original.displaced_volume_m3[:1].copy(),
        model_weight=original.model_weight[:1].copy(),
        material_id=original.material_id[:1].copy(),
    )
    electric_values = np.asarray(
        [[1.0, 0.0], [3.0, 0.0], [1.0, 0.0], [3.0, 0.0]],
        dtype="<f8",
    )
    fields = tuple(
        replace(field, values=electric_values) if field.name == "electric_field" else field
        for field in case.data.fields
    )
    data_path = paths.case_path.with_name("nonuniform-safe.h5")
    info = write(data_path, replace(case.data, fields=fields, sources=(source,)))
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    case_path = paths.case_path.with_name("nonuniform-safe.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    output = tmp_path / "nonuniform-safe-result"
    simulate(load_case(case_path), output)
    result = open_result(output)
    acceleration_x = (
        float(source.charge_number[0]) * 1.602176634e-19 * 2.0 / float(source.mass_kg[0])
    )
    expected_position_x = 0.3 - 0.1 * 0.5 + 0.5 * acceleration_x * 0.5**2
    expected_velocity_x = -0.1 + acceleration_x * 0.5

    np.testing.assert_allclose(
        result.read_final().position_m,
        [[expected_position_x, 0.0]],
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        result.read_final().velocity_m_s,
        [[expected_velocity_x, 0.0]],
        rtol=0.0,
        atol=2.0e-14,
    )
    assert result.manifest["resolved"]["path_kind"] == "rk4_dense"


def test_position_dependent_electric_field_has_fourth_order_public_run(
    tmp_path: Path,
) -> None:
    """Exercise stage-position field sampling through all three public APIs."""

    end_time_s = 0.75
    angular_frequency_s_inv = 2.0
    initial_position_m = np.asarray([0.3, -0.2])
    initial_velocity_m_s = np.asarray([0.0, 0.4])
    phase = angular_frequency_s_inv * end_time_s
    expected_position_x_m = (
        initial_position_m[0] * np.cos(phase)
        + initial_velocity_m_s[0] * np.sin(phase) / angular_frequency_s_inv
    )
    expected_velocity_x_m_s = -initial_position_m[0] * angular_frequency_s_inv * np.sin(
        phase
    ) + initial_velocity_m_s[0] * np.cos(phase)
    position_errors: list[float] = []
    velocity_errors: list[float] = []

    for step_s in (0.125, 0.0625, 0.03125):
        finals = []
        for schedule_name, frame_times in (
            ("no-frames", None),
            ("with-frames", [0.13, 0.47]),
        ):
            case_path = _harmonic_electric_case(
                tmp_path / f"h-{step_s}-{schedule_name}",
                step_s=step_s,
                end_time_s=end_time_s,
                angular_frequency_s_inv=angular_frequency_s_inv,
                initial_position_m=initial_position_m,
                initial_velocity_m_s=initial_velocity_m_s,
                frame_times=frame_times,
            )
            output = tmp_path / f"result-h-{step_s}-{schedule_name}"
            simulate(load_case(case_path), output)
            result = open_result(output)

            assert result.manifest["rk4_enclosure_revision"] == "rk4_global_abs_enclosure_v2"
            assert (
                result.manifest["rk4_dense_path_revision"]
                == "rk4_position_hermite_state_extension_v3"
            )
            assert result.manifest["resolved"]["path_kind"] == "rk4_dense"
            assert result.read_boundary_events().particle_id.size == 0
            finals.append(result.read_final())

        np.testing.assert_array_equal(finals[0].position_m, finals[1].position_m)
        np.testing.assert_array_equal(finals[0].velocity_m_s, finals[1].velocity_m_s)
        np.testing.assert_array_equal(finals[0].charge_number, finals[1].charge_number)
        position_errors.append(abs(float(finals[0].position_m[0, 0]) - expected_position_x_m))
        velocity_errors.append(abs(float(finals[0].velocity_m_s[0, 0]) - expected_velocity_x_m_s))

    for errors in (position_errors, velocity_errors):
        _assert_observed_order(errors, minimum_order=3.5)


@pytest.mark.parametrize("field_layout_kind", ["regular", "p1", "q1"])
def test_general_rk4_material_wall_hit_converges_and_is_schedule_independent(
    tmp_path: Path,
    field_layout_kind: str,
) -> None:
    """A prior wall hit must discard the unsupported tail of an RK4 trial.

    The manufactured motion is

        x'' = -4 x,
        x(0) = 1/2,
        x'(0) = 2,

    so the first right-wall hit is t = atan(3/4) / 2 with x'(t) = 1.
    The field support and material domain both end at x=1.  Every hit step
    therefore has a provisional stage or endpoint beyond field support; the
    run can complete only when first-event localization precedes validity.
    """

    expected_time_s = 0.5 * np.arctan(0.75)
    frame_phase = 2.0 * 0.28
    expected_frame_position_x_m = 0.5 * np.cos(frame_phase) + np.sin(frame_phase)
    expected_frame_velocity_x_m_s = -np.sin(frame_phase) + 2.0 * np.cos(frame_phase)
    time_errors: list[float] = []
    velocity_errors: list[float] = []
    initial_y_m = 0.6 if field_layout_kind == "p1" else 0.5

    for step_s in (0.125, 0.0625, 0.03125):
        results = []
        for schedule_name, frame_times in (
            ("no-frames", None),
            ("with-frames", [0.1, 0.28, 0.4, 0.5]),
        ):
            case_path = _harmonic_electric_case(
                tmp_path / f"wall-{field_layout_kind}-h-{step_s}-{schedule_name}",
                step_s=step_s,
                end_time_s=0.5,
                angular_frequency_s_inv=2.0,
                initial_position_m=np.asarray([0.5, initial_y_m]),
                initial_velocity_m_s=np.asarray([2.0, 0.0]),
                frame_times=frame_times,
                material_wall=True,
                field_layout_kind=field_layout_kind,
            )
            output = tmp_path / f"wall-result-{field_layout_kind}-h-{step_s}-{schedule_name}"

            simulate(load_case(case_path), output)
            result = open_result(output)
            event = result.read_boundary_events()
            final = result.read_final()

            assert result.manifest["resolved"]["path_kind"] == "rk4_dense"
            assert result.manifest["event_refinement"] is not None
            memory_plan = result.manifest["memory_plan"]
            assert isinstance(memory_plan, dict)
            components = memory_plan["components"]
            assert isinstance(components, dict)
            if field_layout_kind == "regular":
                assert components["field_runtime"] == 0
                assert memory_plan["field_preparation_transient_bytes"] == 0
            else:
                assert components["field_runtime"] > np.dtype("<i8").itemsize
                assert memory_plan["field_preparation_transient_bytes"] > 0
            np.testing.assert_array_equal(event.particle_id, [401])
            np.testing.assert_array_equal(event.primary_facet_id, [1])
            np.testing.assert_array_equal(event.candidate_offset, [0, 1])
            np.testing.assert_array_equal(event.candidate_facet_id, [1])
            np.testing.assert_array_equal(event.position_m[:, 1], [initial_y_m])
            np.testing.assert_allclose(
                event.position_m[:, 0],
                [1.0],
                rtol=0.0,
                atol=4.0 * event.position_budget_m[0],
            )
            np.testing.assert_array_equal(event.normal, [[1.0, 0.0]])
            np.testing.assert_array_equal(event.velocity_post_m_s, [[0.0, 0.0]])
            np.testing.assert_array_equal(event.law_id, ["stick"])
            np.testing.assert_array_equal(event.outcome, ["stuck"])
            np.testing.assert_array_equal(final.position_m, event.position_m)
            np.testing.assert_array_equal(final.velocity_m_s, [[0.0, 0.0]])
            np.testing.assert_array_equal(final.lifecycle, [2])
            results.append(result)

        reference_event = results[0].read_boundary_events()
        scheduled_event = results[1].read_boundary_events()
        reference_final = results[0].read_final()
        scheduled_final = results[1].read_final()
        assert results[0].manifest["event_refinement"] == results[1].manifest["event_refinement"]
        assert results[0].manifest["event_refinement"]["maximum_refinement_depth"] <= 24
        np.testing.assert_array_equal(reference_event.time_s, scheduled_event.time_s)
        np.testing.assert_array_equal(
            reference_event.event_ordinal,
            scheduled_event.event_ordinal,
        )
        np.testing.assert_array_equal(reference_event.position_m, scheduled_event.position_m)
        np.testing.assert_array_equal(
            reference_event.velocity_pre_m_s,
            scheduled_event.velocity_pre_m_s,
        )
        np.testing.assert_array_equal(
            reference_event.velocity_post_m_s,
            scheduled_event.velocity_post_m_s,
        )
        np.testing.assert_array_equal(
            reference_event.charge_number_pre,
            scheduled_event.charge_number_pre,
        )
        np.testing.assert_array_equal(
            reference_event.charge_number_post,
            scheduled_event.charge_number_post,
        )
        np.testing.assert_array_equal(
            reference_event.primary_facet_id,
            scheduled_event.primary_facet_id,
        )
        np.testing.assert_array_equal(reference_event.boundary_id, scheduled_event.boundary_id)
        np.testing.assert_array_equal(reference_event.material_id, scheduled_event.material_id)
        np.testing.assert_array_equal(reference_event.normal, scheduled_event.normal)
        np.testing.assert_array_equal(reference_event.model_weight, scheduled_event.model_weight)
        np.testing.assert_array_equal(reference_event.law_id, scheduled_event.law_id)
        np.testing.assert_array_equal(reference_event.outcome, scheduled_event.outcome)
        np.testing.assert_array_equal(
            reference_event.candidate_offset,
            scheduled_event.candidate_offset,
        )
        np.testing.assert_array_equal(
            reference_event.candidate_facet_id,
            scheduled_event.candidate_facet_id,
        )
        np.testing.assert_array_equal(
            reference_event.localization_residual_m,
            scheduled_event.localization_residual_m,
        )
        np.testing.assert_array_equal(
            reference_event.position_budget_m,
            scheduled_event.position_budget_m,
        )
        np.testing.assert_array_equal(
            reference_event.time_budget_s,
            scheduled_event.time_budget_s,
        )
        np.testing.assert_array_equal(reference_final.position_m, scheduled_final.position_m)
        np.testing.assert_array_equal(
            reference_final.velocity_m_s,
            scheduled_final.velocity_m_s,
        )
        np.testing.assert_array_equal(
            reference_final.charge_number,
            scheduled_final.charge_number,
        )
        np.testing.assert_array_equal(reference_final.lifecycle, scheduled_final.lifecycle)

        frames = list(results[1].iter_frames())
        assert [frame.time_s for frame in frames] == [0.1, 0.28, 0.4, 0.5]
        np.testing.assert_allclose(
            frames[1].position_m,
            [[expected_frame_position_x_m, initial_y_m]],
            rtol=0.0,
            atol=2.0e-4,
        )
        np.testing.assert_allclose(
            frames[1].velocity_m_s,
            [[expected_frame_velocity_x_m_s, 0.0]],
            rtol=0.0,
            atol=2.0e-4,
        )
        np.testing.assert_array_equal(frames[1].lifecycle, [1])
        for frame in frames[2:]:
            np.testing.assert_array_equal(frame.position_m, scheduled_final.position_m)
            np.testing.assert_array_equal(frame.velocity_m_s, [[0.0, 0.0]])
            np.testing.assert_array_equal(frame.lifecycle, [2])

        time_errors.append(abs(float(reference_event.time_s[0]) - expected_time_s))
        velocity_errors.append(abs(float(reference_event.velocity_pre_m_s[0, 0]) - 1.0))

    _assert_observed_order(time_errors, minimum_order=3.5)
    assert time_errors[-1] < 2.0e-6
    _assert_observed_order(velocity_errors, minimum_order=3.5)
    assert velocity_errors[-1] < 2.0e-6


@pytest.mark.parametrize("field_layout_kind", ["p1", "q1"])
def test_boundaryless_unstructured_general_rk4_is_rejected(
    tmp_path: Path,
    field_layout_kind: str,
) -> None:
    """An exact-mesh field still needs a continuous boundaryless support proof."""

    case_path = _harmonic_electric_case(
        tmp_path / f"boundaryless-{field_layout_kind}",
        step_s=0.125,
        end_time_s=0.5,
        angular_frequency_s_inv=2.0,
        initial_position_m=np.asarray([0.5, 0.6 if field_layout_kind == "p1" else 0.5]),
        initial_velocity_m_s=np.asarray([2.0, 0.0]),
        frame_times=None,
        field_layout_kind=field_layout_kind,
    )
    output = tmp_path / f"boundaryless-result-{field_layout_kind}"

    with pytest.raises(
        SimulationError,
        match="boundaryless general RK4 requires a fully supported regular field box",
    ):
        simulate(load_case(case_path), output)
    assert not (output / "_SUCCESS").exists()


def test_shared_unresolved_geometry_and_regular_axis_is_prepare_fatal(tmp_path: Path) -> None:
    paths = materialize_microcase("C02", tmp_path / "unresolved-regular-axis")
    case = load_case(paths.case_path)
    lower = 1.0e200
    upper = float(np.nextafter(lower, np.inf))
    geometry_nodes = case.data.geometry.nodes_m.copy()
    original_x = np.unique(geometry_nodes[:, 0])
    assert original_x.size == 2
    geometry_nodes[geometry_nodes[:, 0] == original_x[0], 0] = lower
    geometry_nodes[geometry_nodes[:, 0] == original_x[1], 0] = upper
    geometry = replace(case.data.geometry, nodes_m=geometry_nodes)
    layouts = tuple(
        replace(layout, axis0_m=np.asarray([lower, upper], dtype="<f8"))
        if isinstance(layout, RegularLayout)
        else layout
        for layout in case.data.layouts
    )
    source = replace(
        case.data.sources[0],
        position_m=np.column_stack(
            (
                np.full(case.data.sources[0].particle_id.size, lower, dtype="<f8"),
                case.data.sources[0].position_m[:, 1],
            )
        ),
    )
    data_path = paths.case_path.with_name("unresolved-regular-axis.h5")
    info = write(
        data_path,
        replace(
            case.data,
            geometry=geometry,
            layouts=layouts,
            sources=(source,),
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    case_path = paths.case_path.with_name("unresolved-regular-axis.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(SimulationError, match="volume cell is unresolved at float64 precision"):
        simulate(load_case(case_path), tmp_path / "unresolved-regular-axis-result")


def test_general_rk4_cross_return_finds_first_exit(tmp_path: Path) -> None:
    """An inside endpoint must not hide an earlier outward wall crossing."""

    expected_time_s = 0.5 * np.arctan(0.75)
    case_path = _harmonic_electric_case(
        tmp_path / "wall-cross-return",
        step_s=1.0,
        end_time_s=1.0,
        angular_frequency_s_inv=2.0,
        initial_position_m=np.asarray([0.5, 0.5]),
        initial_velocity_m_s=np.asarray([2.0, 0.0]),
        frame_times=None,
        material_wall=True,
    )
    output = tmp_path / "wall-cross-return-result"

    simulate(load_case(case_path), output)
    result = open_result(output)
    event = result.read_boundary_events()
    final = result.read_final()

    np.testing.assert_array_equal(event.particle_id, [401])
    assert abs(float(event.time_s[0]) - expected_time_s) < 2.0e-5
    np.testing.assert_allclose(
        event.velocity_pre_m_s,
        [[1.0, 0.0]],
        rtol=0.0,
        atol=2.0e-5,
    )
    np.testing.assert_allclose(
        event.position_m,
        [[1.0, 0.5]],
        rtol=0.0,
        atol=event.position_budget_m[0],
    )
    np.testing.assert_array_equal(final.position_m, event.position_m)
    np.testing.assert_array_equal(final.velocity_m_s, [[0.0, 0.0]])
    np.testing.assert_array_equal(final.lifecycle, [2])
    assert result.manifest["event_refinement"]["maximum_refinement_depth"] <= 24


def test_general_rk4_dense_chord_bound_avoids_global_bound_refinement_explosion(
    tmp_path: Path,
) -> None:
    """An unused extreme field cell must not make a transverse hit intractable."""

    angular_frequency_s_inv = 1.0e-6
    amplitude_m = np.hypot(0.5, 2.0 / angular_frequency_s_inv)
    phase_rad = np.atan2(0.5, 2.0 / angular_frequency_s_inv)
    expected_time_s = (np.arcsin(1.0 / amplitude_m) - phase_rad) / angular_frequency_s_inv
    case_path = _loose_global_bound_harmonic_wall_case(tmp_path / "loose-bound")
    output = tmp_path / "loose-bound-result"

    summary = simulate(load_case(case_path), output)
    result = open_result(output)
    event = result.read_boundary_events()
    refinement = result.manifest["event_refinement"]

    assert summary.failure_event_count == 0
    assert result.manifest["resolved"]["path_kind"] == "rk4_dense"
    assert refinement is not None
    assert refinement["candidate_queries"] <= 100
    assert refinement["refinements"] <= 50
    assert refinement["maximum_refinement_depth"] <= 48
    np.testing.assert_array_equal(event.particle_id, [401])
    assert abs(float(event.time_s[0]) - expected_time_s) < 1.0e-10
    np.testing.assert_allclose(
        event.position_m,
        [[1.0, 0.5]],
        rtol=0.0,
        atol=event.position_budget_m[0],
    )


def test_general_rk4_dense_event_bounds_clear_loose_global_root(tmp_path: Path) -> None:
    """A loose global safety enclosure must not subdivide an interior dense path."""

    case_path = _loose_global_bound_harmonic_interior_case(tmp_path / "loose-interior")
    output = tmp_path / "loose-interior-result"

    summary = simulate(load_case(case_path), output)
    result = open_result(output)
    final = result.read_final()
    refinement = result.manifest["event_refinement"]

    assert summary.boundary_event_count == 0
    assert summary.failure_event_count == 0
    assert result.manifest["resolved"]["path_kind"] == "rk4_dense"
    assert refinement == {
        "accepted_particle_pieces": 1,
        "candidate_queries": 1,
        "maximum_refinement_depth": 0,
        "refinements": 0,
    }
    np.testing.assert_allclose(final.position_m, [[0.55, 0.5]], rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(final.velocity_m_s, [[0.1, 0.0]], rtol=0.0, atol=2.0e-13)
    np.testing.assert_array_equal(final.lifecycle, [1])
    np.testing.assert_array_equal(final.kinematics_valid, [1])


def test_general_rk4_specular_hit_advances_the_residual_time(tmp_path: Path) -> None:
    """A curved path must restart at the reflected state and keep its macro target."""

    hit_time_s = 0.5 * np.arctan(0.75)
    residual_s = 0.5 - hit_time_s
    expected_position_x = np.cos(2.0 * residual_s) - 0.5 * np.sin(2.0 * residual_s)
    expected_velocity_x = -2.0 * np.sin(2.0 * residual_s) - np.cos(2.0 * residual_s)
    results = []
    for label in ("no-frames", "with-frames"):
        frame_times = None
        if results:
            resolved_hit_time_s = float(results[0].read_boundary_events().time_s[0])
            frame_times = [0.1, resolved_hit_time_s, 0.4, 0.5]
        case_path = _harmonic_electric_case(
            tmp_path / label,
            step_s=0.5,
            end_time_s=0.5,
            angular_frequency_s_inv=2.0,
            initial_position_m=np.asarray([0.5, 0.5]),
            initial_velocity_m_s=np.asarray([2.0, 0.0]),
            frame_times=frame_times,
            material_wall=True,
        )
        document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
        document["boundaries"] = [
            {
                "boundary_group": "wall",
                "priority": 10,
                "law": "specular",
            }
        ]
        case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        output = tmp_path / f"general-reflection-{label}"
        simulate(load_case(case_path), output)
        results.append(open_result(output))

    event = results[1].read_boundary_events()
    final = results[1].read_final()
    assert event.particle_id.size == 1
    assert abs(float(event.time_s[0]) - hit_time_s) < 2.0e-5
    np.testing.assert_allclose(event.velocity_pre_m_s, [[1.0, 0.0]], rtol=0.0, atol=2.0e-5)
    np.testing.assert_allclose(event.velocity_post_m_s, [[-1.0, 0.0]], rtol=0.0, atol=2.0e-5)
    np.testing.assert_array_equal(event.outcome, ["reflected"])
    np.testing.assert_allclose(
        final.position_m,
        [[expected_position_x, 0.5]],
        rtol=0.0,
        atol=3.0e-4,
    )
    np.testing.assert_allclose(
        final.velocity_m_s,
        [[expected_velocity_x, 0.0]],
        rtol=0.0,
        atol=5.0e-4,
    )
    np.testing.assert_array_equal(results[0].read_boundary_events().time_s, event.time_s)
    np.testing.assert_array_equal(
        results[0].read_boundary_events().velocity_post_m_s, event.velocity_post_m_s
    )
    np.testing.assert_array_equal(results[0].read_final().position_m, final.position_m)
    np.testing.assert_array_equal(results[0].read_final().velocity_m_s, final.velocity_m_s)
    hit_frame = list(results[1].iter_frames())[1]
    np.testing.assert_array_equal(hit_frame.position_m, event.position_m)
    np.testing.assert_array_equal(hit_frame.velocity_m_s, event.velocity_post_m_s)
    assert all(
        result.manifest["boundary_interactions"]["residual_splits"] == 0 for result in results
    )


def test_exponential_midpoint_uses_the_shared_material_event_and_residual_path(
    tmp_path: Path,
) -> None:
    """The native method must not bypass first-hit or post-reflection residual work."""

    results = []
    for label, frame_times in (("no-frames", None), ("with-frames", [0.1, 0.4, 0.5])):
        case_path = _harmonic_electric_case(
            tmp_path / f"exponential-wall-{label}",
            step_s=0.0625,
            end_time_s=0.5,
            angular_frequency_s_inv=2.0,
            initial_position_m=np.asarray([0.5, 0.5]),
            initial_velocity_m_s=np.asarray([2.0, 0.0]),
            frame_times=frame_times,
            material_wall=True,
        )
        document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
        document["solver"]["integrator"] = "exponential_midpoint"
        document["boundaries"] = [
            {
                "boundary_group": "wall",
                "priority": 10,
                "law": "specular",
            }
        ]
        case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        output = tmp_path / f"exponential-wall-result-{label}"
        simulate(load_case(case_path), output)
        results.append(open_result(output))

    expected_hit_time_s = 0.5 * np.arctan(0.75)
    event = results[1].read_boundary_events()
    assert event.particle_id.size == 1
    assert abs(float(event.time_s[0]) - expected_hit_time_s) < 2.0e-3
    np.testing.assert_allclose(event.position_m, [[1.0, 0.5]], rtol=0.0, atol=3.0e-12)
    np.testing.assert_allclose(event.velocity_pre_m_s, [[1.0, 0.0]], rtol=0.0, atol=3.0e-3)
    np.testing.assert_allclose(event.velocity_post_m_s, [[-1.0, 0.0]], rtol=0.0, atol=3.0e-3)
    np.testing.assert_array_equal(event.outcome, ["reflected"])
    assert results[1].manifest["resolved"]["path_kind"] == ("exponential_midpoint_reintegrated")
    refinement = results[1].manifest["event_refinement"]
    assert refinement is not None
    assert refinement["maximum_refinement_depth"] <= 24
    np.testing.assert_array_equal(
        results[0].read_boundary_events().time_s,
        event.time_s,
    )
    np.testing.assert_array_equal(
        results[0].read_final().position_m,
        results[1].read_final().position_m,
    )
    np.testing.assert_array_equal(
        results[0].read_final().velocity_m_s,
        results[1].read_final().velocity_m_s,
    )


def test_general_rk4_surface_departure_returns_to_its_source_facet(tmp_path: Path) -> None:
    """A wall release may depart across a macro boundary and later hit the same facet."""

    release_time_s = 0.1
    expected_hit_time_s = release_time_s + 0.5 * np.pi
    results = []
    for label in ("no-frames", "with-frames"):
        frame_times = None
        if results:
            resolved_hit_time_s = float(results[0].read_boundary_events().time_s[0])
            frame_times = [release_time_s, 0.5, resolved_hit_time_s, 1.8]
        case_path = _general_rk4_surface_case(
            tmp_path / label,
            step_s=0.1,
            end_time_s=1.8,
            release_time_s=release_time_s,
            initial_velocity_m_s=[1.0, 0.0],
            frame_times=frame_times,
        )
        output = tmp_path / f"general-surface-{label}"
        simulate(load_case(case_path), output)
        results.append(open_result(output))

    event = results[1].read_boundary_events()
    final = results[1].read_final()
    assert event.particle_id.size == 1
    assert abs(float(event.time_s[0]) - expected_hit_time_s) < 4.0e-5
    np.testing.assert_allclose(event.position_m, [[0.0, 0.5]], rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(event.velocity_pre_m_s, [[-1.0, 0.0]], rtol=0.0, atol=8.0e-5)
    np.testing.assert_array_equal(event.outcome, ["stuck"])
    np.testing.assert_array_equal(final.position_m, event.position_m)
    np.testing.assert_array_equal(final.velocity_m_s, [[0.0, 0.0]])
    np.testing.assert_array_equal(final.lifecycle, [2])
    np.testing.assert_array_equal(results[0].read_boundary_events().time_s, event.time_s)
    np.testing.assert_array_equal(results[0].read_final().position_m, final.position_m)
    frames = list(results[1].iter_frames())
    np.testing.assert_array_equal(frames[0].position_m, [[0.0, 0.5]])
    np.testing.assert_array_equal(frames[2].position_m, event.position_m)
    np.testing.assert_array_equal(frames[2].lifecycle, [2])
    assert results[1].manifest["resolved"]["path_kind"] == "rk4_dense"


def test_exponential_midpoint_surface_departure_returns_to_its_source_facet(
    tmp_path: Path,
) -> None:
    """The exponential method shares the certified surface-departure path."""

    release_time_s = 0.1
    expected_hit_time_s = release_time_s + 0.5 * np.pi
    case_path = _general_rk4_surface_case(
        tmp_path / "exponential-surface",
        step_s=0.025,
        end_time_s=1.8,
        release_time_s=release_time_s,
        initial_velocity_m_s=[1.0, 0.0],
        frame_times=[release_time_s, 0.5, 1.8],
    )
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["solver"]["integrator"] = "exponential_midpoint"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / "exponential-surface-result"

    simulate(load_case(case_path), output)
    result = open_result(output)
    event = result.read_boundary_events()
    final = result.read_final()

    assert event.particle_id.size == 1
    assert abs(float(event.time_s[0]) - expected_hit_time_s) < 2.0e-3
    np.testing.assert_allclose(event.position_m, [[0.0, 0.5]], rtol=0.0, atol=2.0e-12)
    np.testing.assert_array_equal(event.outcome, ["stuck"])
    np.testing.assert_array_equal(final.position_m, event.position_m)
    np.testing.assert_array_equal(final.velocity_m_s, [[0.0, 0.0]])
    np.testing.assert_array_equal(final.lifecycle, [2])
    assert result.manifest["resolved"]["path_kind"] == "exponential_midpoint_reintegrated"
    assert result.manifest["event_refinement"]["maximum_refinement_depth"] <= 24


def test_general_rk4_tangent_surface_departure_fails_closed(tmp_path: Path) -> None:
    case_path = _general_rk4_surface_case(
        tmp_path / "tangent",
        step_s=0.1,
        end_time_s=0.2,
        release_time_s=0.0,
        initial_velocity_m_s=[0.0, 1.0],
        frame_times=None,
    )
    output = tmp_path / "general-surface-tangent"

    summary = simulate(load_case(case_path), output)
    result = open_result(output)
    failure = result.read_failure_events()
    final = result.read_final()
    reason_code = result.manifest["failure_reason_codes"]["indeterminate_surface_departure"]

    assert summary.failure_event_count == 1
    assert result.manifest["status"] == "complete"
    np.testing.assert_array_equal(failure.particle_id, [1201])
    np.testing.assert_array_equal(failure.time_s, [0.0])
    np.testing.assert_array_equal(failure.reason_code, [reason_code])
    np.testing.assert_array_equal(final.lifecycle, [4])
    np.testing.assert_array_equal(final.failure_reason_code, failure.reason_code)


def test_general_rk4_residual_cap_splits_only_when_another_hit_exists(
    tmp_path: Path,
) -> None:
    gap_m = 2.0**-10
    expected_times_s = np.asarray(
        [2.0**-11 + index * gap_m for index in range(5)],
        dtype=np.float64,
    )
    outputs = []
    for label, interactions, refinements in (
        ("default", 8, 48),
        ("split", 1, 8),
    ):
        case_path = _general_rk4_c09_case(
            tmp_path / label,
            interactions=interactions,
            refinements=refinements,
        )
        output = tmp_path / f"general-c09-{label}"
        simulate(load_case(case_path), output)
        outputs.append(open_result(output))

    default_event = outputs[0].read_boundary_events()
    split_event = outputs[1].read_boundary_events()
    np.testing.assert_allclose(split_event.time_s, expected_times_s, rtol=0.0, atol=2.0e-14)
    for name in (
        "particle_id",
        "event_ordinal",
        "primary_facet_id",
        "candidate_offset",
        "candidate_facet_id",
    ):
        np.testing.assert_array_equal(getattr(default_event, name), getattr(split_event, name))
    for name in ("time_s", "velocity_pre_m_s", "velocity_post_m_s"):
        np.testing.assert_allclose(
            getattr(default_event, name),
            getattr(split_event, name),
            rtol=0.0,
            atol=2.0e-14,
        )
    np.testing.assert_allclose(
        outputs[1].read_final().position_m,
        [[gap_m / 4.0, 64.0 * gap_m]],
        rtol=0.0,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        outputs[1].read_final().velocity_m_s,
        [[-1.0, 0.0]],
        rtol=0.0,
        atol=2.0e-13,
    )
    assert outputs[1].manifest["boundary_interactions"]["residual_splits"] > 0


def test_general_rk4_material_batch_matches_single_particle(tmp_path: Path) -> None:
    """Work partitioning must not change any particle's public result."""

    results = []
    for particle_count in (1, 8):
        case_path = _harmonic_electric_case(
            tmp_path / f"wall-batch-{particle_count}",
            step_s=0.1875,
            end_time_s=0.5,
            angular_frequency_s_inv=2.0,
            initial_position_m=np.asarray([0.5, 0.5]),
            initial_velocity_m_s=np.asarray([2.0, 0.0]),
            frame_times=[0.1, 0.28, 0.4, 0.5],
            material_wall=True,
            particle_count=particle_count,
        )
        output = tmp_path / f"wall-batch-result-{particle_count}"
        simulate(load_case(case_path), output)
        results.append(open_result(output))

    single_event = results[0].read_boundary_events()
    batched_event = results[1].read_boundary_events()
    single_final = results[0].read_final()
    batched_final = results[1].read_final()

    np.testing.assert_array_equal(batched_event.particle_id, np.arange(401, 409))
    for name in (
        "time_s",
        "event_ordinal",
        "position_m",
        "velocity_pre_m_s",
        "velocity_post_m_s",
        "charge_number_pre",
        "charge_number_post",
        "primary_facet_id",
        "boundary_id",
        "material_id",
        "normal",
        "model_weight",
        "law_id",
        "outcome",
        "localization_residual_m",
        "position_budget_m",
        "time_budget_s",
    ):
        expected = np.repeat(getattr(single_event, name), 8, axis=0)
        np.testing.assert_array_equal(getattr(batched_event, name), expected)
    np.testing.assert_array_equal(batched_event.candidate_offset, np.arange(9))
    np.testing.assert_array_equal(
        batched_event.candidate_facet_id,
        np.repeat(single_event.candidate_facet_id, 8),
    )

    for name in ("position_m", "velocity_m_s", "charge_number", "lifecycle"):
        expected = np.repeat(getattr(single_final, name), 8, axis=0)
        np.testing.assert_array_equal(getattr(batched_final, name), expected)

    single_frames = list(results[0].iter_frames())
    batched_frames = list(results[1].iter_frames())
    assert [frame.time_s for frame in batched_frames] == [frame.time_s for frame in single_frames]
    for single_frame, batched_frame in zip(single_frames, batched_frames, strict=True):
        np.testing.assert_array_equal(batched_frame.particle_id, np.arange(401, 409))
        for name in ("position_m", "velocity_m_s", "charge_number", "lifecycle"):
            expected = np.repeat(getattr(single_frame, name), 8, axis=0)
            np.testing.assert_array_equal(getattr(batched_frame, name), expected)

    single_statistics = results[0].manifest["event_refinement"]
    batched_statistics = results[1].manifest["event_refinement"]
    assert single_statistics is not None
    assert batched_statistics is not None
    for name in ("accepted_particle_pieces", "candidate_queries", "refinements"):
        assert batched_statistics[name] == 8 * single_statistics[name]
    assert (
        batched_statistics["maximum_refinement_depth"]
        == single_statistics["maximum_refinement_depth"]
    )


def test_memory_slab_partitioning_does_not_change_public_identity(
    tmp_path: Path,
) -> None:
    """Memory slab size may change execution, not public identity."""

    particle_count = 512
    case_path = _harmonic_electric_case(
        tmp_path / "memory-partition",
        step_s=0.5,
        end_time_s=0.5,
        angular_frequency_s_inv=1.0e-6,
        initial_position_m=np.asarray([0.5, 0.5]),
        initial_velocity_m_s=np.asarray([2.0, 0.0]),
        frame_times=[0.1, 0.3, 0.5],
        material_wall=True,
        particle_count=particle_count,
    )
    case = load_case(case_path)
    fields = tuple(
        replace(
            field,
            values=np.repeat(
                np.asarray([[4.0, -2.0]], dtype="<f8"),
                field.values.shape[0],
                axis=0,
            ),
        )
        if field.name == "electric_field"
        else field
        for field in case.data.fields
    )
    data_path = case_path.with_name("constant-electric.h5")
    info = write(data_path, replace(case.data, fields=fields))
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    wall = document["boundaries"][0]
    wall["law"] = "probabilistic_stick"
    wall["probability"] = 0.5
    wall["otherwise"] = {"law": "specular"}
    document["output"]["probes"] = {
        "particle_ids": [401, 401 + particle_count // 2, 400 + particle_count],
        "schedule": {"explicit_times_s": [0.1, 0.3, 0.5]},
    }

    results = []
    expected_limit_bytes = []
    variants = (
        ("tight", 6),
        ("loose", 64),
    )
    for label, memory_limit_mb in variants:
        variant = {
            **document,
            "resources": {
                **document["resources"],
                "memory_limit_mb": memory_limit_mb,
            },
        }
        variant_path = case_path.with_name(f"{label}.yaml")
        variant_path.write_text(
            yaml.safe_dump(variant, sort_keys=False),
            encoding="utf-8",
        )
        output = tmp_path / f"memory-partition-{label}"
        simulate(load_case(variant_path), output)
        results.append(open_result(output))
        expected_limit_bytes.append(memory_limit_mb * 1024 * 1024)

    plans = [result.manifest["memory_plan"] for result in results]
    assert all(isinstance(plan, dict) for plan in plans)
    assert plans[0]["slab_particles"] < plans[1]["slab_particles"]
    assert plans[1]["slab_particles"] == particle_count
    expected_event_work = 24 * (case.spec.solver.event.max_refinements + 1)
    for plan, limit_bytes in zip(plans, expected_limit_bytes, strict=True):
        phase_peaks = plan["phase_peaks"]
        components = plan["components"]
        assert isinstance(phase_peaks, dict)
        assert isinstance(components, dict)
        assert plan["limit_bytes"] == limit_bytes
        assert plan["planned_bytes"] == max(phase_peaks.values())
        assert phase_peaks["run"] == sum(components.values())
        assert components["slab_proposal_scratch"] == (
            plan["slab_particles"] * plan["scratch_bytes_per_particle"]
        )
        assert components["slab_event_work"] == (
            plan["slab_particles"] * plan["event_work_bytes_per_particle"]
        )
        # The fixture replaces the harmonic field with a uniform electric
        # field, so preparation selects the exact constant-acceleration path.
        assert plan["dense_path_bytes_per_particle"] == 0
        assert components["slab_dense_path"] == 0
        assert plan["certificate_work_bytes_per_particle"] == 0
        assert components["slab_certificate_work"] == 0
        assert plan["event_work_bytes_per_particle"] == expected_event_work
        assert plan["event_staging_capacity"] == plan["event_candidate_capacity"] // 2
        assert plan["slab_particles"] <= plan["event_staging_capacity"]
        assert plan["event_staging_bytes_per_row"] == 490
        assert plan["event_staging_fixed_bytes"] == (
            2 * plan["event_candidate_capacity"] * np.dtype("<i8").itemsize + 16
        )
        assert plan["failure_staging_bytes_per_particle"] == 52
        assert components["slab_event_staging"] == (
            plan["event_staging_fixed_bytes"]
            + plan["slab_particles"] * plan["event_staging_bytes_per_row"]
        )
        assert components["slab_failure_staging"] == (
            plan["slab_particles"] * plan["failure_staging_bytes_per_particle"]
        )
        assert components["geometry_query_scratch"] == (
            3 * plan["event_candidate_capacity"] * np.dtype("<i8").itemsize
        )
        assert plan["planned_bytes"] <= plan["limit_bytes"]

    reference = results[1]
    assert reference.manifest["engine_algorithm_revision"] == "particle_engine_v37"
    assert reference.manifest["compiled_cpu_tile_revision"] == "compiled_cpu_tile_v18"
    assert reference.manifest["geometry_algorithm_revision"] == (
        "line_boundary_stackless_volume_cell_bvh_v5"
    )
    assert plans[1]["revision"] == "solver_owned_memory_plan_v14"
    assert plans[1]["runtime_layout_revision"] == "resident_soa_serial_slab_v6"
    assert reference.manifest["resolved"]["physics_models"]["electric"]["model"] == "coulomb"
    reference_boundary = reference.read_boundary_events()
    assert set(reference_boundary.outcome.tolist()) == {"reflected", "stuck"}
    np.testing.assert_array_equal(
        reference_boundary.particle_id,
        np.arange(401, 401 + particle_count),
    )
    np.testing.assert_array_equal(
        reference_boundary.event_ordinal,
        np.ones(particle_count, dtype="<u4"),
    )
    np.testing.assert_array_equal(
        reference_boundary.candidate_offset, np.arange(particle_count + 1)
    )
    assert reference.read_failure_events().particle_id.size == 0
    assert len(list(reference.iter_frames())) == 3
    assert len(list(reference.iter_probes())) == 3

    assert "threads" not in reference.manifest["requested"]
    assert "threads" not in reference.manifest["resolved"]
    assert "thread_team" not in reference.manifest["resolved"]
    _assert_public_result_identity(reference, results[0])

    hold_document = yaml.safe_load(yaml.safe_dump(document, sort_keys=False))
    hold_wall = hold_document["boundaries"][0]
    hold_wall["law"] = "hold"
    hold_wall.pop("probability")
    hold_wall.pop("otherwise")
    hold_results = []
    for label, memory_limit_mb in variants:
        variant = {
            **hold_document,
            "resources": {
                **hold_document["resources"],
                "memory_limit_mb": memory_limit_mb,
            },
        }
        variant_path = case_path.with_name(f"hold-{label}.yaml")
        variant_path.write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
        output = tmp_path / f"memory-partition-hold-{label}"
        simulate(load_case(variant_path), output)
        hold_results.append(open_result(output))
    for result in hold_results:
        np.testing.assert_array_equal(
            result.read_boundary_events().outcome,
            np.full(particle_count, "held"),
        )
        np.testing.assert_array_equal(
            result.read_final().lifecycle,
            np.full(particle_count, 5, dtype="<u1"),
        )
    _assert_public_result_identity(hold_results[1], hold_results[0])


def test_general_rk4_wall_ambiguity_fails_when_refinement_budget_is_exhausted(
    tmp_path: Path,
) -> None:
    particle_count = 8
    case_path = _harmonic_electric_case(
        tmp_path / "wall-refinement-budget",
        step_s=0.1875,
        end_time_s=0.5,
        angular_frequency_s_inv=2.0,
        initial_position_m=np.asarray([0.5, 0.5]),
        initial_velocity_m_s=np.asarray([2.0, 0.0]),
        frame_times=None,
        material_wall=True,
        particle_count=particle_count,
    )
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["solver"]["event"]["max_refinements"] = 1
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / "wall-refinement-budget-result"
    summary = simulate(load_case(case_path), output)
    assert summary.failure_event_count == particle_count

    reference = open_result(output)
    failure = reference.read_failure_events()
    final = reference.read_final()
    reason_code = reference.manifest["failure_reason_codes"]["indeterminate_event"]
    assert reference.manifest["status"] == "complete"
    np.testing.assert_array_equal(
        failure.particle_id,
        np.arange(401, 401 + particle_count),
    )
    np.testing.assert_array_equal(
        failure.reason_code,
        np.full(particle_count, reason_code),
    )
    np.testing.assert_array_equal(final.lifecycle, np.full(particle_count, 4))
    np.testing.assert_array_equal(final.failure_reason_code, failure.reason_code)


def test_general_rk4_enclosure_uses_particle_local_release_duration(tmp_path: Path) -> None:
    paths = materialize_microcase("C02", tmp_path / "midstep-release")
    case = load_case(paths.case_path)
    original = case.data.sources[0]
    source = replace(
        original,
        release_time_s=np.asarray([0.1], dtype="<f8"),
    )
    data_path = paths.case_path.with_name("midstep-release.h5")
    info = write(data_path, replace(case.data, sources=(source,)))
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["output"]["trajectories"] = None
    case_path = paths.case_path.with_name("midstep-release.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    output = tmp_path / "midstep-release-result"
    simulate(load_case(case_path), output)
    final = open_result(output).read_final()
    duration_s = 0.9
    tau_s = 0.5
    target_velocity = np.asarray([0.1, -0.05])
    decay = np.exp(-duration_s / tau_s)
    response = 1.0 - decay
    expected_position = (
        source.position_m[0]
        + target_velocity * duration_s
        + tau_s * response * (source.velocity_m_s[0] - target_velocity)
    )
    expected_velocity = target_velocity + decay * (source.velocity_m_s[0] - target_velocity)

    np.testing.assert_allclose(final.position_m, [expected_position], rtol=0.0, atol=8.0e-5)
    np.testing.assert_allclose(final.velocity_m_s, [expected_velocity], rtol=0.0, atol=1.6e-4)


def test_continuous_epstein_applicability_is_output_schedule_independent(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C02", tmp_path / "epstein-applicability")
    case = load_case(paths.case_path)
    original = case.data.sources[0]
    source = replace(
        original,
        particle_id=np.asarray([int(original.particle_id[0]), int(original.particle_id[0]) + 1]),
        release_time_s=np.repeat(original.release_time_s, 2),
        position_m=np.repeat(original.position_m, 2, axis=0),
        velocity_m_s=np.asarray([[60.0, 0.0], original.velocity_m_s[0]], dtype="<f8"),
        charge_number=np.repeat(original.charge_number, 2),
        mass_kg=np.repeat(original.mass_kg, 2),
        drag_diameter_m=np.repeat(original.drag_diameter_m, 2),
        electrostatic_radius_m=np.repeat(original.electrostatic_radius_m, 2),
        displaced_volume_m3=np.repeat(original.displaced_volume_m3, 2),
        model_weight=np.repeat(original.model_weight, 2),
        material_id=np.repeat(original.material_id, 2),
    )
    data_path = paths.case_path.with_name("epstein-applicability.h5")
    info = write(data_path, replace(case.data, sources=(source,)))

    finals = []
    for label, frame_times in (("no-frames", None), ("with-frame", [5.0e-4])):
        document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
        document["case"]["data_path"] = data_path.name
        document["case"]["expected_content_hash"] = info.content_hash
        document["time"] = {"start_s": 0.0, "end_s": 1.0e-3, "dt_s": 1.0e-3}
        if frame_times is None:
            document["output"]["trajectories"] = None
        else:
            document["output"]["trajectories"]["schedule"] = {"explicit_times_s": frame_times}
        case_path = paths.case_path.with_name(f"epstein-applicability-{label}.yaml")
        case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        output = tmp_path / f"epstein-applicability-{label}-result"

        simulate(load_case(case_path), output)
        result = open_result(output)
        final = result.read_final()
        failure = result.read_failure_events()
        reason = result.manifest["failure_reason_codes"]["model_applicability"]
        np.testing.assert_array_equal(final.lifecycle, [4, 1])
        np.testing.assert_array_equal(final.failure_reason_code, [reason, 0])
        np.testing.assert_array_equal(failure.particle_id, [source.particle_id[0]])
        np.testing.assert_array_equal(failure.reason_code, [reason])
        finals.append(final)

    np.testing.assert_array_equal(finals[0].position_m, finals[1].position_m)
    np.testing.assert_array_equal(finals[0].velocity_m_s, finals[1].velocity_m_s)
    np.testing.assert_array_equal(finals[0].lifecycle, finals[1].lifecycle)


def test_boundaryless_quadratic_excursion_is_local_failure_with_or_without_frames(
    tmp_path: Path,
) -> None:
    """Sparse point samples must not certify continuous support for an exact parabola."""

    paths = materialize_microcase("C05", tmp_path / "support-excursion")
    case = load_case(paths.case_path)
    original = case.data.sources[0]
    source = replace(
        original,
        particle_id=original.particle_id[:1].copy(),
        release_time_s=np.asarray([0.0], dtype="<f8"),
        position_m=np.asarray([[-0.9, 0.5]], dtype="<f8"),
        velocity_m_s=np.asarray([[-1.8, 0.0]], dtype="<f8"),
        charge_number=original.charge_number[:1].copy(),
        mass_kg=original.mass_kg[:1].copy(),
        drag_diameter_m=original.drag_diameter_m[:1].copy(),
        electrostatic_radius_m=original.electrostatic_radius_m[:1].copy(),
        displaced_volume_m3=np.asarray([0.0], dtype="<f8"),
        model_weight=original.model_weight[:1].copy(),
        material_id=original.material_id[:1].copy(),
    )
    data_path = paths.case_path.with_name("support-excursion.h5")
    info = write(data_path, replace(case.data, sources=(source,)))

    for label, frame_times in (("no-frames", None), ("with-frame", [0.5])):
        document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
        document["case"]["data_path"] = data_path.name
        document["case"]["expected_content_hash"] = info.content_hash
        document["time"] = {"start_s": 0.0, "end_s": 1.0, "dt_s": 1.0}
        document["physics"]["gravity_buoyancy"]["gravity_m_s2"] = [7.2, 0.0]
        if frame_times is None:
            document["output"]["trajectories"] = None
        else:
            document["output"]["trajectories"]["schedule"] = {"explicit_times_s": frame_times}
        case_path = paths.case_path.with_name(f"support-excursion-{label}.yaml")
        case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        output = tmp_path / f"support-excursion-{label}-result"

        simulate(load_case(case_path), output)
        result = open_result(output)
        final = result.read_final()
        reason = result.manifest["failure_reason_codes"]["field_support"]
        np.testing.assert_array_equal(final.lifecycle, [4])
        np.testing.assert_array_equal(final.failure_reason_code, [reason])
        np.testing.assert_array_equal(result.read_failure_events().reason_code, [reason])


def test_boundaryless_quadratic_overflow_does_not_poison_normal_neighbor(
    tmp_path: Path,
) -> None:
    paths = materialize_microcase("C05", tmp_path / "quadratic-overflow-neighbor")
    case = load_case(paths.case_path)
    original = case.data.sources[0]
    source = replace(
        original,
        particle_id=np.asarray([501, 502], dtype="<i8"),
        release_time_s=np.zeros(2, dtype="<f8"),
        position_m=np.asarray([[0.0, 0.5], [0.0, 0.5]], dtype="<f8"),
        velocity_m_s=np.asarray([[0.0, 0.0], [1.0e308, 0.0]], dtype="<f8"),
        charge_number=np.repeat(original.charge_number[:1], 2),
        mass_kg=np.repeat(original.mass_kg[:1], 2),
        drag_diameter_m=np.repeat(original.drag_diameter_m[:1], 2),
        electrostatic_radius_m=np.repeat(original.electrostatic_radius_m[:1], 2),
        displaced_volume_m3=np.zeros(2, dtype="<f8"),
        model_weight=np.repeat(original.model_weight[:1], 2),
        material_id=np.repeat(original.material_id[:1], 2),
    )
    data_path = paths.case_path.with_name("quadratic-overflow-neighbor.h5")
    info = write(data_path, replace(case.data, sources=(source,)))
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"] = {"start_s": 0.0, "end_s": 2.0, "dt_s": 2.0}
    document["physics"]["gravity_buoyancy"]["gravity_m_s2"] = [0.1, 0.0]
    document["output"]["trajectories"] = None
    case_path = paths.case_path.with_name("quadratic-overflow-neighbor.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    output = tmp_path / "quadratic-overflow-neighbor-result"
    simulate(load_case(case_path), output)
    result = open_result(output)
    final = result.read_final()
    failure = result.read_failure_events()
    reason = result.manifest["failure_reason_codes"]["nonfinite_physics"]

    np.testing.assert_array_equal(final.particle_id, [501, 502])
    np.testing.assert_array_equal(final.lifecycle, [1, 4])
    np.testing.assert_array_equal(final.failure_reason_code, [0, reason])
    np.testing.assert_array_equal(final.position_m[0], [0.2, 0.5])
    np.testing.assert_array_equal(final.velocity_m_s[0], [0.2, 0.0])
    np.testing.assert_array_equal(failure.particle_id, [502])
    np.testing.assert_array_equal(failure.reason_code, [reason])


def test_regular_nonuniform_excursion_is_local_failure_independent_of_output_schedule(
    tmp_path: Path,
) -> None:
    """Reject a case whose full RK4 samples hide an unsupported shortened path.

    The full-step x samples/end remain in [-1, 1], while a step to 0.87 h ends
    below -1.  A requested frame must therefore not decide whether the run is valid.
    """

    paths = materialize_microcase("C04", tmp_path / "regular-nonuniform")
    case = load_case(paths.case_path)
    original = case.data.sources[0]
    position_m = np.asarray([[0.7178049036, 0.0]], dtype="<f8")
    velocity_m_s = np.asarray([[-7.0408375495, 0.0]], dtype="<f8")
    source = replace(
        original,
        particle_id=original.particle_id[:1].copy(),
        release_time_s=original.release_time_s[:1].copy(),
        position_m=position_m,
        velocity_m_s=velocity_m_s,
        charge_number=original.charge_number[:1].copy(),
        mass_kg=original.mass_kg[:1].copy(),
        drag_diameter_m=original.drag_diameter_m[:1].copy(),
        electrostatic_radius_m=original.electrostatic_radius_m[:1].copy(),
        displaced_volume_m3=original.displaced_volume_m3[:1].copy(),
        model_weight=original.model_weight[:1].copy(),
        material_id=original.material_id[:1].copy(),
    )
    slope_s2_inverse = -42.7669775174
    offset_m_s2 = 0.2780151248
    charge_to_acceleration = (
        float(source.charge_number[0]) * 1.602176634e-19 / float(source.mass_kg[0])
    )
    node_x_m = np.asarray([-1.0, -1.0, 1.0, 1.0], dtype="<f8")
    electric_x_V_m = (slope_s2_inverse * node_x_m + offset_m_s2) / charge_to_acceleration
    electric_values = np.column_stack((electric_x_V_m, np.zeros(electric_x_V_m.size, dtype="<f8")))
    fields = tuple(
        replace(field, values=electric_values) if field.name == "electric_field" else field
        for field in case.data.fields
    )
    data_path = paths.case_path.with_name("regular-nonuniform.h5")
    info = write(data_path, replace(case.data, fields=fields, sources=(source,)))
    step_s = 0.2911182108

    for label, frame_times in (("no-frames", None), ("with-frame", [0.87 * step_s])):
        document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
        document["case"]["data_path"] = data_path.name
        document["case"]["expected_content_hash"] = info.content_hash
        document["time"] = {"start_s": 0.0, "end_s": step_s, "dt_s": step_s}
        if frame_times is None:
            document["output"]["trajectories"] = None
        else:
            document["output"]["trajectories"]["schedule"] = {"explicit_times_s": frame_times}
        case_path = paths.case_path.with_name(f"regular-nonuniform-{label}.yaml")
        case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        output = tmp_path / f"regular-nonuniform-{label}-result"

        simulate(load_case(case_path), output)
        result = open_result(output)
        final = result.read_final()
        reason = result.manifest["failure_reason_codes"]["field_support"]
        np.testing.assert_array_equal(final.lifecycle, [4])
        np.testing.assert_array_equal(final.failure_reason_code, [reason])
        np.testing.assert_array_equal(result.read_failure_events().reason_code, [reason])


def test_constant_acceleration_wall_hit_is_exact_across_dt_and_output_schedule(
    tmp_path: Path,
) -> None:
    expected_time_s = 1.0 - 1.0 / np.sqrt(2.0)
    results = []
    for label, dt_s, frame_times in (
        ("single-step", 2.0, [0.0, 0.25, 0.5, 2.0]),
        ("four-steps", 0.5, [0.0, 0.25, 0.5, 2.0]),
        ("eight-steps", 0.25, [0.0, 0.25, 0.5, 2.0]),
        ("no-frames", 2.0, None),
    ):
        case_path = _constant_gravity_wall_case(
            tmp_path / label,
            dt_s=dt_s,
            frame_times=frame_times,
        )
        output = tmp_path / f"{label}-result"
        simulate(load_case(case_path), output)
        result = open_result(output)
        event = result.read_boundary_events()
        final = result.read_final()

        assert event.particle_id.size == 1
        assert abs(event.time_s[0] - expected_time_s) <= 4.0 * event.time_budget_s[0]
        np.testing.assert_allclose(
            event.position_m,
            [[1.0, 0.5]],
            rtol=0.0,
            atol=4.0 * event.position_budget_m[0],
        )
        np.testing.assert_allclose(
            event.velocity_pre_m_s,
            [[np.sqrt(2.0), 0.0]],
            rtol=0.0,
            atol=2.0e-12,
        )
        np.testing.assert_array_equal(event.velocity_post_m_s, [[0.0, 0.0]])
        np.testing.assert_array_equal(event.outcome, ["stuck"])
        np.testing.assert_array_equal(final.position_m, event.position_m)
        np.testing.assert_array_equal(final.velocity_m_s, [[0.0, 0.0]])
        np.testing.assert_array_equal(final.lifecycle, [2])
        assert result.manifest["resolved"]["path_kind"] == "quadratic_exact"
        results.append(result)

    reference_event = results[0].read_boundary_events()
    reference_final = results[0].read_final()
    for result in results[1:]:
        event = result.read_boundary_events()
        final = result.read_final()
        np.testing.assert_allclose(event.time_s, reference_event.time_s, rtol=0.0, atol=2.0e-15)
        np.testing.assert_array_equal(event.position_m, reference_event.position_m)
        np.testing.assert_allclose(
            event.velocity_pre_m_s,
            reference_event.velocity_pre_m_s,
            rtol=0.0,
            atol=4.0e-15,
        )
        np.testing.assert_array_equal(final.position_m, reference_final.position_m)
        np.testing.assert_array_equal(final.velocity_m_s, reference_final.velocity_m_s)
        np.testing.assert_array_equal(final.lifecycle, reference_final.lifecycle)

    assert list(results[-1].iter_frames()) == []
    frames = list(results[0].iter_frames())
    assert [frame.time_s for frame in frames] == [0.0, 0.25, 0.5, 2.0]
    np.testing.assert_allclose(frames[1].position_m, [[0.9375, 0.5]], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(frames[1].velocity_m_s, [[1.5, 0.0]], rtol=0.0, atol=0.0)
    for frame in frames[2:]:
        np.testing.assert_array_equal(frame.position_m, [[1.0, 0.5]])
        np.testing.assert_array_equal(frame.velocity_m_s, [[0.0, 0.0]])
        np.testing.assert_array_equal(frame.lifecycle, [2])


def test_hold_retains_curved_impact_state_after_exact_quadratic_hit(tmp_path: Path) -> None:
    case_path = _constant_gravity_wall_case(
        tmp_path / "hold-curved-hit",
        dt_s=2.0,
        frame_times=[0.0, 0.25, 0.5, 2.0],
        wall_law="hold",
    )
    output = tmp_path / "hold-curved-hit-result"
    simulate(load_case(case_path), output)
    result = open_result(output)
    event = result.read_boundary_events()
    final = result.read_final()

    expected_time_s = 1.0 - 1.0 / np.sqrt(2.0)
    expected_velocity_m_s = np.asarray([[np.sqrt(2.0), 0.0]])
    assert event.particle_id.size == 1
    assert abs(event.time_s[0] - expected_time_s) <= 4.0 * event.time_budget_s[0]
    np.testing.assert_allclose(
        event.position_m,
        [[1.0, 0.5]],
        rtol=0.0,
        atol=4.0 * event.position_budget_m[0],
    )
    np.testing.assert_allclose(event.velocity_pre_m_s, expected_velocity_m_s, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(event.velocity_post_m_s, expected_velocity_m_s, rtol=0.0, atol=2e-12)
    np.testing.assert_array_equal(event.law_id, ["hold"])
    np.testing.assert_array_equal(event.outcome, ["held"])
    np.testing.assert_allclose(final.position_m, [[1.0, 0.5]], rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(final.velocity_m_s, expected_velocity_m_s, rtol=0.0, atol=2e-12)
    np.testing.assert_array_equal(final.lifecycle, [5])
    np.testing.assert_array_equal(final.kinematics_valid, [1])

    frames = list(result.iter_frames())
    np.testing.assert_array_equal(frames[1].lifecycle, [1])
    for frame in frames[2:]:
        np.testing.assert_allclose(frame.position_m, [[1.0, 0.5]], rtol=0.0, atol=2e-15)
        np.testing.assert_allclose(frame.velocity_m_s, expected_velocity_m_s, rtol=0.0, atol=2e-12)
        np.testing.assert_array_equal(frame.lifecycle, [5])


def test_constant_acceleration_surface_uses_the_first_nonzero_normal_derivative(
    tmp_path: Path,
) -> None:
    case_path = _constant_acceleration_surface_case(
        tmp_path / "acceleration-departure",
        acceleration_m_s2=[2.0, 0.0],
        initial_velocity_m_s=[0.0, 0.0],
        collector_law="stick",
        frame_times=[0.0, 0.25, 0.5],
    )
    output = tmp_path / "acceleration-departure-result"

    simulate(load_case(case_path), output)
    result = open_result(output)
    final = result.read_final()
    frames = list(result.iter_frames())

    assert result.read_boundary_events().particle_id.size == 0
    np.testing.assert_array_equal(final.position_m, [[0.25, 0.5]])
    np.testing.assert_array_equal(final.velocity_m_s, [[1.0, 0.0]])
    np.testing.assert_array_equal(final.lifecycle, [1])
    np.testing.assert_array_equal(
        [frame.position_m[0] for frame in frames],
        [[0.0, 0.5], [0.0625, 0.5], [0.25, 0.5]],
    )
    np.testing.assert_array_equal(
        [frame.velocity_m_s[0] for frame in frames],
        [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
    )
    assert result.manifest["engine_algorithm_revision"] == "particle_engine_v37"
    assert result.manifest["compiled_cpu_tile_revision"] == "compiled_cpu_tile_v18"
    assert result.manifest["event_algorithm_revision"] == "line_quadratic_rk4_axis_first_hit_v16"
    assert result.manifest["resolved"]["path_kind"] == "quadratic_exact"


def test_acceleration_departure_certificate_survives_release_at_a_macro_end(
    tmp_path: Path,
) -> None:
    case_path = _constant_acceleration_surface_case(
        tmp_path / "macro-end-release",
        acceleration_m_s2=[2.0, 0.0],
        initial_velocity_m_s=[0.0, 0.0],
        collector_law="stick",
        frame_times=[0.5, 0.75, 1.0],
        release_time_s=0.5,
        end_time_s=1.0,
    )
    output = tmp_path / "macro-end-release-result"

    simulate(load_case(case_path), output)
    result = open_result(output)
    frames = list(result.iter_frames())

    assert result.read_boundary_events().particle_id.size == 0
    np.testing.assert_array_equal(result.read_final().position_m, [[0.25, 0.5]])
    np.testing.assert_array_equal(
        [frame.position_m[0] for frame in frames],
        [[0.0, 0.5], [0.0625, 0.5], [0.25, 0.5]],
    )


def test_acceleration_departure_certificate_persists_until_another_wall_hit(
    tmp_path: Path,
) -> None:
    case_path = _constant_acceleration_surface_case(
        tmp_path / "small-macro-steps",
        acceleration_m_s2=[2.0, 0.0],
        initial_velocity_m_s=[0.0, 0.0],
        collector_law="stick",
        frame_times=None,
        end_time_s=3.0e-7,
        dt_s=1.0e-7,
    )
    output = tmp_path / "small-macro-steps-result"

    simulate(load_case(case_path), output)
    result = open_result(output)

    assert result.read_boundary_events().particle_id.size == 0
    np.testing.assert_allclose(
        result.read_final().position_m,
        [[9.0e-14, 0.5]],
        rtol=0.0,
        atol=2.0e-29,
    )
    np.testing.assert_allclose(
        result.read_final().velocity_m_s,
        [[6.0e-7, 0.0]],
        rtol=0.0,
        atol=2.0e-22,
    )


def test_constant_acceleration_surface_outward_acceleration_is_an_immediate_stick(
    tmp_path: Path,
) -> None:
    case_path = _constant_acceleration_surface_case(
        tmp_path / "acceleration-impact",
        acceleration_m_s2=[-2.0, 0.0],
        initial_velocity_m_s=[0.0, 0.0],
        collector_law="stick",
        frame_times=None,
    )
    output = tmp_path / "acceleration-impact-result"

    simulate(load_case(case_path), output)
    result = open_result(output)
    event = result.read_boundary_events()
    final = result.read_final()

    np.testing.assert_array_equal(event.time_s, [0.0])
    np.testing.assert_array_equal(event.position_m, [[0.0, 0.5]])
    np.testing.assert_array_equal(event.velocity_pre_m_s, [[0.0, 0.0]])
    np.testing.assert_array_equal(event.outcome, ["stuck"])
    np.testing.assert_array_equal(final.position_m, [[0.0, 0.5]])
    np.testing.assert_array_equal(final.lifecycle, [2])


def test_indeterminate_boundary_policy_fails_only_the_affected_particle(
    tmp_path: Path,
) -> None:
    case_path = _constant_acceleration_surface_case(
        tmp_path / "unresolved-acceleration-impact",
        acceleration_m_s2=[-2.0, 0.0],
        initial_velocity_m_s=[0.0, 0.0],
        collector_law="specular",
        frame_times=None,
        include_companion=True,
    )
    output = tmp_path / "unresolved-acceleration-impact-result"

    summary = simulate(load_case(case_path), output)
    result = open_result(output)
    failure = result.read_failure_events()
    final = result.read_final()

    assert summary.failure_event_count == 1
    assert result.manifest["status"] == "complete"
    np.testing.assert_array_equal(failure.particle_id, [1101])
    np.testing.assert_array_equal(failure.time_s, [0.0])
    assert int(failure.reason_code[0]) > 0
    reason_code = result.manifest["failure_reason_codes"]["indeterminate_boundary_policy"]
    np.testing.assert_array_equal(failure.reason_code, [reason_code])
    np.testing.assert_array_equal(final.particle_id, [501, 1101])
    np.testing.assert_array_equal(final.lifecycle, [1, 4])
    np.testing.assert_array_equal(final.kinematics_valid, [1, 0])
    np.testing.assert_array_equal(final.failure_reason_code, [0, failure.reason_code[0]])


def test_constant_acceleration_surface_immediate_reflection_keeps_the_residual_time(
    tmp_path: Path,
) -> None:
    results = []
    for label, frame_times in (
        ("no-frames", None),
        ("with-frames", [0.0, 0.25, 0.5]),
    ):
        case_path = _constant_acceleration_surface_case(
            tmp_path / label,
            acceleration_m_s2=[2.0, 0.0],
            initial_velocity_m_s=[-1.0, 0.0],
            collector_law="specular",
            frame_times=frame_times,
        )
        output = tmp_path / f"{label}-result"
        simulate(load_case(case_path), output)
        results.append(open_result(output))

    event = results[1].read_boundary_events()
    final = results[1].read_final()
    np.testing.assert_array_equal(event.time_s, [0.0])
    np.testing.assert_array_equal(event.velocity_pre_m_s, [[-1.0, 0.0]])
    np.testing.assert_array_equal(event.velocity_post_m_s, [[1.0, 0.0]])
    np.testing.assert_array_equal(event.outcome, ["reflected"])
    np.testing.assert_array_equal(final.position_m, [[0.75, 0.5]])
    np.testing.assert_array_equal(final.velocity_m_s, [[2.0, 0.0]])
    np.testing.assert_array_equal(
        results[0].read_boundary_events().velocity_post_m_s,
        event.velocity_post_m_s,
    )
    np.testing.assert_array_equal(results[0].read_final().position_m, final.position_m)
    frames = list(results[1].iter_frames())
    np.testing.assert_array_equal(
        [frame.velocity_m_s[0] for frame in frames],
        [[1.0, 0.0], [1.5, 0.0], [2.0, 0.0]],
    )


@pytest.mark.parametrize("case_id", ["C02", "C03", "C04", "C05"])
def test_force_coupled_axisymmetric_rz_matches_cartesian_away_from_axis(
    tmp_path: Path, case_id: str
) -> None:
    paths = materialize_microcase(case_id, tmp_path / f"rz-force-{case_id}")
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    gravity = document["physics"].get("gravity_buoyancy")
    if gravity is not None:
        gravity["gravity_m_s2"][0] = 0.0
        paths.case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    case = load_case(paths.case_path)
    radial_shift = np.asarray([1.5, 0.0])
    geometry = replace(case.data.geometry, nodes_m=case.data.geometry.nodes_m + radial_shift)
    layouts = tuple(
        replace(layout, axis0_m=layout.axis0_m + radial_shift[0]) for layout in case.data.layouts
    )
    fields = tuple(
        replace(field, components=("r", "z"), stored_basis="axisymmetric_rz")
        if len(field.components) == 2
        else field
        for field in case.data.fields
    )
    sources = tuple(
        replace(source, position_m=source.position_m + radial_shift) for source in case.data.sources
    )
    data_path = paths.case_path.with_name("rz-force.h5")
    info = write(
        data_path,
        replace(
            case.data,
            coordinate_system="axisymmetric_rz",
            geometry=geometry,
            layouts=layouts,
            fields=fields,
            sources=sources,
        ),
    )
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["motion"]["mode"] = "axisymmetric_rz_meridional"
    case_path = paths.case_path.with_name("rz-force.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    simulate(case, tmp_path / "xy-force-result")
    simulate(load_case(case_path), tmp_path / "rz-force-result")
    xy = open_result(tmp_path / "xy-force-result")
    rz = open_result(tmp_path / "rz-force-result")

    np.testing.assert_allclose(
        rz.read_final().position_m - radial_shift,
        xy.read_final().position_m,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        rz.read_final().velocity_m_s,
        xy.read_final().velocity_m_s,
        rtol=0.0,
        atol=1.0e-15,
    )
    np.testing.assert_array_equal(rz.read_final().charge_number, xy.read_final().charge_number)
    assert rz.manifest["resolved"]["path_kind"] == "rk4_dense"
    assert rz.manifest["physics_catalog_revision"] == "inertial_langevin_rz_catalog_v17"


def test_epstein_rz_axis_crossing_folds_and_continues_the_residual_time(tmp_path: Path) -> None:
    signed_radius = -0.3 + 0.4 * np.exp(-0.4)
    signed_velocity = -0.8 * np.exp(-0.4)
    position_errors = []
    velocity_errors = []
    results = []
    for index, dt_s in enumerate((0.05, 0.025, 0.0125)):
        case_path = _epstein_rz_axis_case(
            tmp_path / f"dt-{index}",
            dt_s=dt_s,
            frame_times=[0.0, 0.1, 0.15, 0.2],
        )
        output = tmp_path / f"result-dt-{index}"
        simulate(load_case(case_path), output)
        result = open_result(output)
        final = result.read_final()
        position_errors.append(abs(float(final.position_m[0, 0]) - abs(signed_radius)))
        velocity_errors.append(abs(float(final.velocity_m_s[0, 0]) - abs(signed_velocity)))
        assert result.read_boundary_events().particle_id.size == 0
        assert result.manifest["boundary_interactions"] == {
            "wall_events": 0,
            "residual_splits": 0,
            "axis_crossings": 1,
        }
        for frame in result.iter_frames():
            assert frame.position_m[0, 0] >= 0.0
        results.append(result)

    _assert_observed_order(position_errors, minimum_order=3.5)
    _assert_observed_order(velocity_errors, minimum_order=3.5)

    no_frames_path = _epstein_rz_axis_case(
        tmp_path / "no-frames",
        dt_s=0.0125,
        frame_times=None,
    )
    simulate(load_case(no_frames_path), tmp_path / "result-no-frames")
    without_frames = open_result(tmp_path / "result-no-frames")
    np.testing.assert_array_equal(
        results[-1].read_final().position_m, without_frames.read_final().position_m
    )
    np.testing.assert_array_equal(
        results[-1].read_final().velocity_m_s, without_frames.read_final().velocity_m_s
    )
    assert results[-1].manifest["event_refinement"] == without_frames.manifest["event_refinement"]
    assert (
        results[-1].manifest["boundary_interactions"]
        == without_frames.manifest["boundary_interactions"]
    )


def test_epstein_rz_axis_precedes_the_later_material_wall(tmp_path: Path) -> None:
    case_path = _epstein_rz_axis_case(
        tmp_path / "axis-then-wall",
        dt_s=0.05,
        frame_times=None,
        outer_wall=True,
    )
    simulate(load_case(case_path), tmp_path / "axis-then-wall-result")
    result = open_result(tmp_path / "axis-then-wall-result")
    event = result.read_boundary_events()

    np.testing.assert_allclose(event.time_s, [-0.5 * np.log(0.25)], atol=3.0e-6)
    np.testing.assert_allclose(event.position_m, [[0.2, 0.0]], atol=3.0e-12)
    np.testing.assert_allclose(event.velocity_pre_m_s, [[0.2, 0.0]], atol=6.0e-6)
    np.testing.assert_array_equal(event.event_ordinal, [1])
    np.testing.assert_array_equal(event.outcome, ["stuck"])
    assert result.manifest["boundary_interactions"]["axis_crossings"] == 1
    assert result.manifest["boundary_interactions"]["wall_events"] == 1


def test_exponential_midpoint_rz_axis_precedes_the_later_material_wall(
    tmp_path: Path,
) -> None:
    """RZ folding and wall arbitration share the same curved-path work loop."""

    case_path = _epstein_rz_axis_case(
        tmp_path / "exponential-axis-then-wall",
        dt_s=0.8,
        frame_times=[0.1, 0.4, 0.8],
        outer_wall=True,
    )
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["solver"]["integrator"] = "exponential_midpoint"
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    simulate(load_case(case_path), tmp_path / "exponential-axis-then-wall-result")
    result = open_result(tmp_path / "exponential-axis-then-wall-result")
    event = result.read_boundary_events()

    np.testing.assert_allclose(event.time_s, [-0.5 * np.log(0.25)], atol=3.0e-11)
    np.testing.assert_allclose(event.position_m, [[0.2, 0.0]], atol=3.0e-12)
    np.testing.assert_allclose(event.velocity_pre_m_s, [[0.2, 0.0]], atol=6.0e-11)
    np.testing.assert_array_equal(event.event_ordinal, [1])
    np.testing.assert_array_equal(event.outcome, ["stuck"])
    assert result.manifest["resolved"]["path_kind"] == ("exponential_midpoint_reintegrated")
    assert result.manifest["boundary_interactions"]["axis_crossings"] == 1
    assert result.manifest["boundary_interactions"]["wall_events"] == 1


@pytest.mark.parametrize("integrator", ["rk4_fixed", "exponential_midpoint"])
def test_finite_speed_epstein_reuses_rz_axis_and_material_wall_path(
    tmp_path: Path,
    integrator: str,
) -> None:
    case_path = _epstein_rz_axis_case(
        tmp_path / f"finite-speed-axis-{integrator}",
        dt_s=0.025,
        frame_times=[0.1, 0.4, 0.8],
        outer_wall=True,
        finite_speed=True,
    )
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["solver"]["integrator"] = integrator
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / f"finite-speed-axis-result-{integrator}"

    simulate(load_case(case_path), output)
    result = open_result(output)
    event = result.read_boundary_events()

    assert event.particle_id.size == 1
    np.testing.assert_allclose(event.position_m, [[0.2, 0.0]], atol=3.0e-12)
    np.testing.assert_array_equal(event.outcome, ["stuck"])
    assert result.manifest["boundary_interactions"]["axis_crossings"] == 1
    assert result.manifest["boundary_interactions"]["wall_events"] == 1
    assert result.manifest["resolved"]["physics_models"]["drag"]["model"] == (
        "epstein_finite_speed"
    )


@pytest.mark.parametrize("support_reaches_axis", [False, True], ids=["annular", "axis-reaching"])
def test_rz_radial_gravity_is_rejected_for_annular_and_axis_reaching_support(
    tmp_path: Path, support_reaches_axis: bool
) -> None:
    paths = materialize_microcase("C05", tmp_path / "rz-radial-gravity")
    case = load_case(paths.case_path)
    shift = np.asarray([1.5, 0.0])
    data_path = paths.case_path.with_name("rz-radial-gravity.h5")
    info = write(
        data_path,
        replace(
            case.data,
            coordinate_system="axisymmetric_rz",
            geometry=replace(case.data.geometry, nodes_m=case.data.geometry.nodes_m + shift),
            layouts=tuple(
                replace(
                    layout,
                    axis0_m=(
                        np.asarray([0.0, float(layout.axis0_m[-1] + shift[0])])
                        if support_reaches_axis
                        else layout.axis0_m + shift[0]
                    ),
                )
                for layout in case.data.layouts
            ),
            sources=tuple(
                replace(source, position_m=source.position_m + shift)
                for source in case.data.sources
            ),
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["motion"]["mode"] = "axisymmetric_rz_meridional"
    document["physics"]["gravity_buoyancy"]["gravity_m_s2"] = [1.0, -10.0]
    case_path = paths.case_path.with_name("rz-radial-gravity.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(SimulationError, match=r"requires gravity_m_s2\[0\] = 0"):
        simulate(load_case(case_path), tmp_path / "rz-radial-gravity-result")


def test_epstein_rz_axis_state_is_invariant_without_radial_motion(tmp_path: Path) -> None:
    case_path = _epstein_rz_axis_case(
        tmp_path / "axis-invariant",
        dt_s=0.2,
        frame_times=[0.0, 0.1, 0.2],
        initial_position_m=(0.0, 0.0),
        initial_velocity_m_s=(0.0, 0.3),
    )
    simulate(load_case(case_path), tmp_path / "axis-invariant-result")
    result = open_result(tmp_path / "axis-invariant-result")

    np.testing.assert_array_equal(result.read_final().position_m[:, 0], [0.0])
    np.testing.assert_array_equal(result.read_final().velocity_m_s[:, 0], [0.0])
    assert result.manifest["boundary_interactions"]["axis_crossings"] == 0
    assert result.manifest["event_refinement"]["refinements"] == 0
    for frame in result.iter_frames():
        np.testing.assert_array_equal(frame.position_m[:, 0], [0.0])
        np.testing.assert_array_equal(frame.velocity_m_s[:, 0], [0.0])


def test_rz_skewed_p1_axis_state_is_invariant_away_from_slanted_walls(tmp_path: Path) -> None:
    paths = materialize_microcase("C04", tmp_path / "rz-skewed-p1-axis")
    case = load_case(paths.case_path)
    nodes = np.asarray([[0.1, 0.3], [0.0, 1.1], [0.0, 0.2]], dtype="<f8")
    connectivity = np.asarray([[0, 1, 2]], dtype="<i8")
    geometry = GeometryData(
        nodes_m=nodes,
        boundary=BoundaryData(
            line2=np.asarray([[0, 1], [2, 0]], dtype="<i8"),
            boundary_id=np.asarray([10, 10], dtype="<i4"),
            group_id=np.asarray([0, 0], dtype="<i4"),
            material_id=np.asarray([0, 0], dtype="<i4"),
            owner_cell_type=np.asarray([1, 1], dtype="<u1"),
            owner_cell_local_index=np.asarray([0, 0], dtype="<i8"),
            orientation=np.asarray([1, 1], dtype="<i1"),
        ),
        group_names=("wall",),
        tri3=connectivity,
        tri3_domain_id=np.asarray([0], dtype="<i4"),
    )
    layout = P1TriLayout(
        "p1",
        nodes.copy(),
        connectivity.copy(),
        np.asarray([1], dtype="<u1"),
    )
    field = replace(
        case.data.fields[0],
        layout=layout.name,
        components=("r", "z"),
        stored_basis="axisymmetric_rz",
        values=np.asarray([[4.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype="<f8"),
    )
    original = case.data.sources[0]
    source = replace(
        original,
        particle_id=original.particle_id[:1].copy(),
        release_time_s=original.release_time_s[:1].copy(),
        position_m=np.asarray([[0.0, 0.317]], dtype="<f8"),
        velocity_m_s=np.zeros((1, 2), dtype="<f8"),
        charge_number=original.charge_number[:1].copy(),
        mass_kg=original.mass_kg[:1].copy(),
        drag_diameter_m=original.drag_diameter_m[:1].copy(),
        electrostatic_radius_m=original.electrostatic_radius_m[:1].copy(),
        displaced_volume_m3=original.displaced_volume_m3[:1].copy(),
        model_weight=original.model_weight[:1].copy(),
        material_id=original.material_id[:1].copy(),
    )
    data_path = paths.case_path.with_name("rz-skewed-p1-axis.h5")
    info = write(
        data_path,
        replace(
            case.data,
            coordinate_system="axisymmetric_rz",
            geometry=geometry,
            layouts=(layout,),
            fields=(field,),
            sources=(source,),
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["motion"]["mode"] = "axisymmetric_rz_meridional"
    document["time"] = {"start_s": 0.0, "end_s": 0.01, "dt_s": 0.01}
    document["boundaries"] = [{"boundary_group": "wall", "priority": 10, "law": "stick"}]
    document["output"]["trajectories"] = None
    case_path = paths.case_path.with_name("rz-skewed-p1-axis.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    simulate(load_case(case_path), tmp_path / "rz-skewed-p1-axis-result")
    result = open_result(tmp_path / "rz-skewed-p1-axis-result")

    np.testing.assert_array_equal(result.read_final().position_m, [[0.0, 0.317]])
    np.testing.assert_array_equal(result.read_final().velocity_m_s, [[0.0, 0.0]])
    assert result.manifest["event_refinement"]["refinements"] == 0
    assert result.manifest["boundary_interactions"]["axis_crossings"] == 0
    assert result.read_boundary_events().particle_id.size == 0


def test_epstein_unstable_fixed_step_is_rejected_at_prepare(tmp_path: Path) -> None:
    paths = materialize_microcase("C03", tmp_path / "C03-unstable")
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["time"]["dt_s"] = document["time"]["end_s"]
    case_path = paths.case_path.with_name("unstable.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / "unstable-result"

    with pytest.raises(SimulationError, match="drag_velocity_lipschitz"):
        simulate(load_case(case_path), output)
    assert not (output / "_SUCCESS").exists()


@pytest.mark.parametrize(
    ("integrator", "minimum_order"),
    [("rk4_fixed", 3.5), ("exponential_midpoint", 1.8)],
)
def test_finite_speed_epstein_converges_with_both_production_integrators(
    tmp_path: Path,
    integrator: str,
    minimum_order: float,
) -> None:
    end_time_s = 1.0e-3
    reference_position, reference_velocity = _finite_speed_epstein_reference(end_time_s)
    errors: list[float] = []
    for divisions in (4, 8, 16, 32):
        case_path = _finite_speed_epstein_case(
            tmp_path / f"{integrator}-{divisions}",
            dt_s=end_time_s / divisions,
            integrator=integrator,
        )
        output = tmp_path / f"finite-speed-{integrator}-{divisions}"
        simulate(load_case(case_path), output)
        result = open_result(output)
        final = result.read_final()
        error = max(
            abs(float(final.position_m[0, 0]) - reference_position),
            abs(float(final.velocity_m_s[0, 0]) - reference_velocity),
        )
        errors.append(error)
        assert result.manifest["resolved"]["physics_models"]["drag"] == {
            "model": "epstein_finite_speed",
            "revision": "epstein_finite_speed_maxwell_mixed_equal_temperature_v1",
        }
        expected_path = (
            "rk4_dense" if integrator == "rk4_fixed" else "exponential_midpoint_reintegrated"
        )
        assert result.manifest["resolved"]["path_kind"] == expected_path
        if integrator == "exponential_midpoint":
            assert result.manifest["memory_plan"]["certificate_work_bytes_per_particle"] == 544

    assert errors[0] > errors[1] > errors[2] > errors[3]
    observed_orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert min(observed_orders[-2:]) >= minimum_order


@pytest.mark.parametrize("dt_s", [0.75, 0.375, 0.125])
def test_exponential_midpoint_c03_matches_the_closed_form(
    tmp_path: Path,
    dt_s: float,
) -> None:
    """The native method is exact for constant linear drag and additive force."""

    paths = materialize_microcase("C03", tmp_path / f"C03-exponential-{dt_s}")
    expected = json.loads(paths.expected_path.read_text(encoding="utf-8"))
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["solver"]["integrator"] = "exponential_midpoint"
    document["time"]["dt_s"] = dt_s
    case_path = paths.case_path.with_name("exponential.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    output = tmp_path / f"C03-exponential-result-{dt_s}"
    simulate(load_case(case_path), output)
    result = open_result(output)
    final = result.read_final()
    frames = list(result.iter_frames())

    np.testing.assert_allclose(final.position_m, expected["position_m"][-1], rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(
        final.velocity_m_s, expected["velocity_m_s"][-1], rtol=0.0, atol=1e-13
    )
    np.testing.assert_array_equal(final.charge_number, expected["charge_number"][-1])
    assert [frame.time_s for frame in frames] == expected["time_s"]
    for index, frame in enumerate(frames):
        np.testing.assert_allclose(
            frame.position_m,
            expected["position_m"][index],
            rtol=0.0,
            atol=1.0e-13,
        )
        np.testing.assert_allclose(
            frame.velocity_m_s,
            expected["velocity_m_s"][index],
            rtol=0.0,
            atol=1.0e-13,
        )
        np.testing.assert_array_equal(frame.charge_number, expected["charge_number"][index])
    assert result.manifest["requested"]["integrator"] == "exponential_midpoint"
    assert result.manifest["resolved"]["integrator"] == "exponential_midpoint"
    assert result.manifest["resolved"]["path_kind"] == "exponential_midpoint_reintegrated"
    assert result.manifest["exponential_midpoint_revision"] == (
        "charge_stable_exponential_midpoint_v3"
    )
    assert (
        result.manifest["exponential_midpoint_enclosure_revision"]
        == "exponential_midpoint_local_stage_enclosure_v4"
    )
    assert result.manifest["maximum_dt_over_tau"] == pytest.approx(4.0 * dt_s)


def test_exponential_midpoint_result_is_independent_of_output_schedule(tmp_path: Path) -> None:
    paths = materialize_microcase("C03", tmp_path / "C03-exponential-schedule")
    dense = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    dense["solver"]["integrator"] = "exponential_midpoint"
    dense["time"]["dt_s"] = dense["time"]["end_s"]
    no_frames = yaml.safe_load(yaml.safe_dump(dense, sort_keys=False))
    no_frames["output"]["trajectories"] = None
    dense_path = paths.case_path.with_name("exponential-dense.yaml")
    no_frames_path = paths.case_path.with_name("exponential-no-frames.yaml")
    dense_path.write_text(yaml.safe_dump(dense, sort_keys=False), encoding="utf-8")
    no_frames_path.write_text(yaml.safe_dump(no_frames, sort_keys=False), encoding="utf-8")

    simulate(load_case(dense_path), tmp_path / "exponential-dense-result")
    simulate(load_case(no_frames_path), tmp_path / "exponential-no-frames-result")
    dense_result = open_result(tmp_path / "exponential-dense-result")
    no_frames_result = open_result(tmp_path / "exponential-no-frames-result")

    np.testing.assert_array_equal(
        dense_result.read_final().position_m,
        no_frames_result.read_final().position_m,
    )
    np.testing.assert_array_equal(
        dense_result.read_final().velocity_m_s,
        no_frames_result.read_final().velocity_m_s,
    )
    np.testing.assert_array_equal(
        dense_result.read_final().charge_number,
        no_frames_result.read_final().charge_number,
    )


def _assert_public_result_identity(reference: Any, partitioned: Any) -> None:
    reference_revisions = {
        name: value for name, value in reference.manifest.items() if name.endswith("_revision")
    }
    partitioned_revisions = {
        name: value for name, value in partitioned.manifest.items() if name.endswith("_revision")
    }
    assert reference_revisions == partitioned_revisions
    for name in (
        "status",
        "result_schema_version",
        "case_name",
        "data_content_hash",
        "case_schema_version",
        "data_coordinate_system",
        "motion_mode",
        "time",
        "event",
        "random_draw_kinds",
        "event_refinement",
        "boundary_interactions",
        "source_id_to_name",
        "lifecycle_counts",
        "failure_reason_codes",
        "failure_reason_counts",
        "maximum_dt_over_tau",
        "maximum_dt_charge_lipschitz",
        "counts",
    ):
        assert reference.manifest[name] == partitioned.manifest[name]
    for name in ("integrator", "backend", "seed"):
        assert reference.manifest["requested"][name] == partitioned.manifest["requested"][name]
    for name in (
        "integrator",
        "backend",
        "path_kind",
        "physics_models",
        "sources",
        "boundary_laws",
        "required_fields",
    ):
        assert reference.manifest["resolved"][name] == partitioned.manifest["resolved"][name]

    reader_names = (
        "read_final",
        "read_release_events",
        "read_boundary_events",
        "read_failure_events",
        "read_lifecycle_series",
    )
    reference_records = [getattr(reference, name)() for name in reader_names]
    partitioned_records = [getattr(partitioned, name)() for name in reader_names]
    for iterator_name in ("iter_frames", "iter_probes"):
        reference_items = list(getattr(reference, iterator_name)())
        partitioned_items = list(getattr(partitioned, iterator_name)())
        assert len(reference_items) == len(partitioned_items)
        reference_records.extend(reference_items)
        partitioned_records.extend(partitioned_items)

    for reference_record, partitioned_record in zip(
        reference_records,
        partitioned_records,
        strict=True,
    ):
        assert type(reference_record) is type(partitioned_record)
        for field in dataclass_fields(reference_record):
            reference_value = getattr(reference_record, field.name)
            partitioned_value = getattr(partitioned_record, field.name)
            if not isinstance(reference_value, np.ndarray):
                assert reference_value == partitioned_value
                continue
            assert isinstance(partitioned_value, np.ndarray)
            assert reference_value.dtype == partitioned_value.dtype
            assert reference_value.shape == partitioned_value.shape
            np.testing.assert_array_equal(reference_value, partitioned_value)
            assert reference_value.tobytes(order="C") == partitioned_value.tobytes(order="C")


def _finite_speed_epstein_case(
    directory: Path,
    *,
    dt_s: float,
    integrator: str,
) -> Path:
    paths = materialize_microcase("C02", directory)
    case = load_case(paths.case_path)
    molecular_mass_kg = 4.65e-26
    temperature_K = 300.0
    most_probable_speed = np.sqrt(2.0 * 1.380649e-23 * temperature_K / molecular_mass_kg)
    fields = tuple(
        replace(field, values=np.zeros_like(field.values))
        if field.name == "gas_velocity"
        else replace(field, values=250.0 * field.values)
        if field.name == "gas_density"
        else field
        for field in case.data.fields
    )
    source = replace(
        case.data.sources[0],
        position_m=np.asarray([[0.0, 0.0]], dtype="<f8"),
        velocity_m_s=np.asarray([[most_probable_speed, 0.0]], dtype="<f8"),
    )
    data_path = paths.case_path.with_name("finite-speed.h5")
    info = write(data_path, replace(case.data, fields=fields, sources=(source,)))
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"] = {"start_s": 0.0, "end_s": 1.0e-3, "dt_s": dt_s}
    document["solver"]["integrator"] = integrator
    document["physics"]["drag"] = {
        "model": "epstein_finite_speed",
        "revision": "epstein_finite_speed_maxwell_mixed_equal_temperature_v1",
        "gas_velocity_field": "gas_velocity",
        "gas_density_field": "gas_density",
        "gas_temperature_field": "gas_temperature",
        "gas_mean_free_path_field": "gas_mean_free_path",
        "gas_molecular_mass_kg": molecular_mass_kg,
        "diffuse_reflection_fraction": 0.6,
        "maximum_speed_ratio": 3.0,
        "applicability": "error",
    }
    document["output"]["trajectories"] = None
    case_path = paths.case_path.with_name("finite-speed.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _finite_speed_epstein_reference(end_time_s: float) -> tuple[float, float]:
    molecular_mass_kg = 4.65e-26
    temperature_K = 300.0
    speed_scale = np.sqrt(2.0 * 1.380649e-23 * temperature_K / molecular_mass_kg)
    diffuse_factor = 0.6 * np.pi / 8.0

    def derivative(state: np.ndarray) -> np.ndarray:
        speed_ratio = abs(float(state[1])) / speed_scale
        if speed_ratio <= 0.1:
            square = speed_ratio * speed_ratio
            specular_factor = 1.0 + square * (
                1.0 / 5.0 + square * (-1.0 / 70.0 + square * (1.0 / 630.0 - square / 5544.0))
            )
        else:
            drag_coefficient = (2.0 * speed_ratio * speed_ratio + 1.0) * np.exp(
                -(speed_ratio * speed_ratio)
            ) / (np.sqrt(np.pi) * speed_ratio**3) + (
                4.0 * speed_ratio**4 + 4.0 * speed_ratio * speed_ratio - 1.0
            ) * math.erf(speed_ratio) / (2.0 * speed_ratio**4)
            specular_factor = 3.0 * np.sqrt(np.pi) * speed_ratio * drag_coefficient / 16.0
        rate = 500.0 * (specular_factor + diffuse_factor)
        return np.asarray([state[1], -rate * state[1]], dtype="<f8")

    step_count = 32768
    step = end_time_s / step_count
    state = np.asarray([0.0, speed_scale], dtype="<f8")
    for _ in range(step_count):
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * step * k1)
        k3 = derivative(state + 0.5 * step * k2)
        k4 = derivative(state + step * k3)
        state += step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return float(state[0]), float(state[1])


def _epstein_rz_axis_case(
    directory: Path,
    *,
    dt_s: float,
    frame_times: list[float] | None,
    outer_wall: bool = False,
    initial_position_m: tuple[float, float] = (0.1, 0.0),
    initial_velocity_m_s: tuple[float, float] = (-0.8, 0.0),
    finite_speed: bool = False,
) -> Path:
    paths = materialize_microcase("C02", directory)
    case = load_case(paths.case_path)
    radial_shift = np.asarray([1.0, 0.0])
    if outer_wall:
        geometry = GeometryData(
            nodes_m=np.asarray([[0.0, -1.0], [0.2, -1.0], [0.2, 1.0], [0.0, 1.0]]),
            boundary=BoundaryData(
                line2=np.asarray([[0, 1], [1, 2], [2, 3]], dtype="<i8"),
                boundary_id=np.asarray([10, 10, 10], dtype="<i4"),
                group_id=np.asarray([0, 0, 0], dtype="<i4"),
                material_id=np.asarray([0, 0, 0], dtype="<i4"),
                owner_cell_type=np.asarray([2, 2, 2], dtype="<u1"),
                owner_cell_local_index=np.asarray([0, 0, 0], dtype="<i8"),
                orientation=np.asarray([1, 1, 1], dtype="<i1"),
            ),
            group_names=("collector",),
            quad4=np.asarray([[0, 1, 2, 3]], dtype="<i8"),
            quad4_domain_id=np.asarray([0], dtype="<i4"),
        )
        layouts = tuple(
            replace(
                layout,
                axis0_m=np.asarray([0.0, 0.2], dtype="<f8"),
                axis1_m=np.asarray([-1.0, 1.0], dtype="<f8"),
            )
            for layout in case.data.layouts
        )
    else:
        geometry = replace(case.data.geometry, nodes_m=case.data.geometry.nodes_m + radial_shift)
        layouts = tuple(
            replace(layout, axis0_m=layout.axis0_m + radial_shift[0])
            for layout in case.data.layouts
        )
    fields = tuple(
        replace(
            field,
            components=("r", "z"),
            stored_basis="axisymmetric_rz",
            values=np.zeros_like(field.values),
        )
        if field.name == "gas_velocity"
        else field
        for field in case.data.fields
    )
    source = replace(
        case.data.sources[0],
        position_m=np.asarray([initial_position_m], dtype="<f8"),
        velocity_m_s=np.asarray([initial_velocity_m_s], dtype="<f8"),
    )
    data_path = paths.case_path.with_name("rz-axis.h5")
    info = write(
        data_path,
        replace(
            case.data,
            coordinate_system="axisymmetric_rz",
            geometry=geometry,
            layouts=layouts,
            fields=fields,
            sources=(source,),
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["motion"]["mode"] = "axisymmetric_rz_meridional"
    if finite_speed:
        linear_drag = document["physics"]["drag"]
        document["physics"]["drag"] = {
            "model": "epstein_finite_speed",
            "revision": "epstein_finite_speed_maxwell_mixed_equal_temperature_v1",
            "gas_velocity_field": linear_drag["gas_velocity_field"],
            "gas_density_field": linear_drag["gas_density_field"],
            "gas_temperature_field": linear_drag["gas_temperature_field"],
            "gas_mean_free_path_field": linear_drag["gas_mean_free_path_field"],
            "gas_molecular_mass_kg": linear_drag["gas_molecular_mass_kg"],
            "diffuse_reflection_fraction": 0.0,
            "maximum_speed_ratio": 1.0,
            "applicability": "error",
        }
    document["time"] = {
        "start_s": 0.0,
        "end_s": 0.8 if outer_wall else 0.2,
        "dt_s": dt_s,
    }
    document["boundaries"] = (
        [{"boundary_group": "collector", "priority": 10, "law": "stick"}] if outer_wall else []
    )
    document["output"]["trajectories"] = (
        None
        if frame_times is None
        else {"selection": "all", "schedule": {"explicit_times_s": frame_times}}
    )
    case_path = paths.case_path.with_name("rz-axis.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _assert_observed_order(errors: list[float], *, minimum_order: float) -> None:
    assert errors[0] > errors[1] > errors[2]
    orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert min(orders) >= minimum_order


def _constant_gravity_wall_case(
    directory: Path,
    *,
    dt_s: float,
    frame_times: list[float] | None,
    wall_law: str = "stick",
) -> Path:
    force_paths = materialize_microcase("C05", directory / "force")
    wall_paths = materialize_microcase("C07", directory / "wall")
    force_case = load_case(force_paths.case_path)
    wall_case = load_case(wall_paths.case_path)
    original = force_case.data.sources[0]
    source = replace(
        original,
        particle_id=original.particle_id[:1].copy(),
        release_time_s=np.asarray([0.0], dtype="<f8"),
        position_m=np.asarray([[0.5, 0.5]], dtype="<f8"),
        velocity_m_s=np.asarray([[2.0, 0.0]], dtype="<f8"),
        charge_number=original.charge_number[:1].copy(),
        mass_kg=original.mass_kg[:1].copy(),
        drag_diameter_m=original.drag_diameter_m[:1].copy(),
        electrostatic_radius_m=original.electrostatic_radius_m[:1].copy(),
        displaced_volume_m3=np.asarray([0.0], dtype="<f8"),
        model_weight=original.model_weight[:1].copy(),
        material_id=original.material_id[:1].copy(),
    )
    data_path = force_paths.case_path.with_name("force-wall.h5")
    info = write(
        data_path,
        replace(
            force_case.data,
            geometry=wall_case.data.geometry,
            sources=(source,),
        ),
    )
    document = yaml.safe_load(force_paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"] = {"start_s": 0.0, "end_s": 2.0, "dt_s": dt_s}
    document["physics"]["gravity_buoyancy"]["gravity_m_s2"] = [-2.0, 0.0]
    document["boundaries"] = [{"boundary_group": "wall", "priority": 10, "law": wall_law}]
    if frame_times is None:
        document["output"]["trajectories"] = None
    else:
        document["output"]["trajectories"]["schedule"] = {"explicit_times_s": frame_times}
    case_path = force_paths.case_path.with_name("force-wall.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _constant_acceleration_surface_case(
    directory: Path,
    *,
    acceleration_m_s2: list[float],
    initial_velocity_m_s: list[float],
    collector_law: str,
    frame_times: list[float] | None,
    release_time_s: float = 0.0,
    end_time_s: float = 0.5,
    dt_s: float = 0.5,
    include_companion: bool = False,
) -> Path:
    force_paths = materialize_microcase("C05", directory / "force")
    wall_paths = materialize_microcase("C08", directory / "wall")
    force_case = load_case(force_paths.case_path)
    wall_case = load_case(wall_paths.case_path)
    particle = force_case.data.sources[0]
    companion = replace(
        particle,
        particle_id=np.asarray([501], dtype="<i8"),
        release_time_s=np.asarray([0.0], dtype="<f8"),
        position_m=np.asarray([[0.75, 0.5]], dtype="<f8"),
        velocity_m_s=np.zeros((1, 2), dtype="<f8"),
        charge_number=particle.charge_number[:1].copy(),
        mass_kg=particle.mass_kg[:1].copy(),
        drag_diameter_m=particle.drag_diameter_m[:1].copy(),
        electrostatic_radius_m=particle.electrostatic_radius_m[:1].copy(),
        displaced_volume_m3=np.zeros(1, dtype="<f8"),
        model_weight=particle.model_weight[:1].copy(),
        material_id=particle.material_id[:1].copy(),
    )
    data_path = force_paths.case_path.with_name("surface-force.h5")
    info = write(
        data_path,
        replace(
            force_case.data,
            geometry=wall_case.data.geometry,
            sources=(companion,) if include_companion else force_case.data.sources,
        ),
    )
    document = yaml.safe_load(force_paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"] = {"start_s": 0.0, "end_s": end_time_s, "dt_s": dt_s}
    document["physics"]["gravity_buoyancy"]["gravity_m_s2"] = acceleration_m_s2
    document["sources"] = [
        {
            "name": "surface_release",
            "type": "surface",
            "boundary_group": "collector",
            "count": 1,
            "particle_id_start": 1101,
            "particle": {
                "charge_number": 0.0,
                "mass_kg": float(particle.mass_kg[0]),
                "drag_diameter_m": float(particle.drag_diameter_m[0]),
                "electrostatic_radius_m": float(particle.electrostatic_radius_m[0]),
                "displaced_volume_m3": 0.0,
                "model_weight": 1.0,
                "material_id": int(particle.material_id[0]),
            },
            "position": {"model": "edge_fraction", "fraction": 0.5},
            "velocity": {"model": "fixed", "value_m_s": initial_velocity_m_s},
            "release": {"model": "fixed", "time_s": release_time_s},
        }
    ]
    if include_companion:
        document["sources"].append(
            {"name": "interior_companion", "type": "table", "table": companion.name}
        )
    collector: dict[str, object] = {
        "boundary_group": "collector",
        "priority": 10,
        "law": collector_law,
    }
    document["boundaries"] = [
        collector,
        {"boundary_group": "mirror", "priority": 10, "law": "stick"},
        {"boundary_group": "caps", "priority": 10, "law": "escape"},
    ]
    if frame_times is None:
        document["output"]["trajectories"] = None
    else:
        document["output"]["trajectories"] = {
            "selection": "all",
            "schedule": {"explicit_times_s": frame_times},
        }
    case_path = force_paths.case_path.with_name("surface-force.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _general_rk4_surface_case(
    directory: Path,
    *,
    step_s: float,
    end_time_s: float,
    release_time_s: float,
    initial_velocity_m_s: list[float],
    frame_times: list[float] | None,
) -> Path:
    force_path = _harmonic_electric_case(
        directory / "force",
        step_s=step_s,
        end_time_s=end_time_s,
        angular_frequency_s_inv=2.0,
        initial_position_m=np.asarray([0.25, 0.5]),
        initial_velocity_m_s=np.asarray([0.0, 0.0]),
        frame_times=None,
    )
    wall_paths = materialize_microcase("C08", directory / "wall")
    force_case = load_case(force_path)
    wall_case = load_case(wall_paths.case_path)
    particle = force_case.data.sources[0]
    data_path = force_path.with_name("general-surface.h5")
    info = write(
        data_path,
        replace(
            force_case.data,
            geometry=wall_case.data.geometry,
        ),
    )
    document = yaml.safe_load(force_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"] = {"start_s": 0.0, "end_s": end_time_s, "dt_s": step_s}
    document["sources"] = [
        {
            "name": "general_surface_release",
            "type": "surface",
            "boundary_group": "collector",
            "count": 1,
            "particle_id_start": 1201,
            "particle": {
                "charge_number": float(particle.charge_number[0]),
                "mass_kg": float(particle.mass_kg[0]),
                "drag_diameter_m": float(particle.drag_diameter_m[0]),
                "electrostatic_radius_m": float(particle.electrostatic_radius_m[0]),
                "displaced_volume_m3": float(particle.displaced_volume_m3[0]),
                "model_weight": 1.0,
                "material_id": int(particle.material_id[0]),
            },
            "position": {"model": "edge_fraction", "fraction": 0.5},
            "velocity": {"model": "fixed", "value_m_s": initial_velocity_m_s},
            "release": {"model": "fixed", "time_s": release_time_s},
        }
    ]
    document["boundaries"] = [
        {"boundary_group": "collector", "priority": 10, "law": "stick"},
        {"boundary_group": "mirror", "priority": 10, "law": "stick"},
        {"boundary_group": "caps", "priority": 10, "law": "escape"},
    ]
    if frame_times is None:
        document["output"]["trajectories"] = None
    else:
        document["output"]["trajectories"] = {
            "selection": "all",
            "schedule": {"explicit_times_s": frame_times},
        }
    case_path = force_path.with_name("general-surface.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _general_rk4_c09_case(
    directory: Path,
    *,
    interactions: int,
    refinements: int,
) -> Path:
    force_path = _harmonic_electric_case(
        directory / "force",
        step_s=0.01,
        end_time_s=0.01,
        angular_frequency_s_inv=1.0,
        initial_position_m=np.asarray([0.0, 0.0]),
        initial_velocity_m_s=np.asarray([0.0, 0.0]),
        frame_times=None,
    )
    c09_paths = materialize_microcase("C09", directory / "wall")
    force_case = load_case(force_path)
    c09_case = load_case(c09_paths.case_path)
    force_particle = force_case.data.sources[0]
    c09_particle = c09_case.data.sources[0]
    source = replace(
        c09_particle,
        charge_number=force_particle.charge_number[:1].copy(),
        mass_kg=force_particle.mass_kg[:1].copy(),
        drag_diameter_m=force_particle.drag_diameter_m[:1].copy(),
        electrostatic_radius_m=force_particle.electrostatic_radius_m[:1].copy(),
        displaced_volume_m3=force_particle.displaced_volume_m3[:1].copy(),
        model_weight=force_particle.model_weight[:1].copy(),
        material_id=force_particle.material_id[:1].copy(),
    )
    layout = force_case.data.layouts[0]
    if not isinstance(layout, RegularLayout):
        raise TypeError("general C09 reference requires a regular field layout")
    axis0_m = layout.axis0_m
    axis1_m = layout.axis1_m
    node_y_m = np.tile(axis1_m, axis0_m.size)
    center_y_m = float(source.position_m[0, 1])
    transverse_frequency_s_inv = 1.0e-6
    charge_to_acceleration = (
        float(source.charge_number[0]) * 1.602176634e-19 / float(source.mass_kg[0])
    )
    electric_values = np.column_stack(
        (
            np.zeros(node_y_m.size, dtype="<f8"),
            -(transverse_frequency_s_inv**2) * (node_y_m - center_y_m) / charge_to_acceleration,
        )
    )
    fields = tuple(
        replace(field, values=electric_values) if field.name == "electric_field" else field
        for field in force_case.data.fields
    )
    data_path = force_path.with_name("general-c09.h5")
    info = write(
        data_path,
        replace(
            force_case.data,
            geometry=c09_case.data.geometry,
            fields=fields,
            sources=(source,),
        ),
    )
    document = yaml.safe_load(force_path.read_text(encoding="utf-8"))
    c09_document = yaml.safe_load(c09_paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"] = c09_document["time"]
    document["boundaries"] = c09_document["boundaries"]
    document["solver"]["event"]["max_interactions_per_step"] = interactions
    document["solver"]["event"]["max_refinements"] = refinements
    document["output"]["trajectories"] = None
    case_path = force_path.with_name("general-c09.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _harmonic_electric_case(
    directory: Path,
    *,
    step_s: float,
    end_time_s: float,
    angular_frequency_s_inv: float,
    initial_position_m: np.ndarray,
    initial_velocity_m_s: np.ndarray,
    frame_times: list[float] | None,
    material_wall: bool = False,
    particle_count: int = 1,
    field_layout_kind: str = "regular",
) -> Path:
    paths = materialize_microcase("C04", directory)
    case = load_case(paths.case_path)
    geometry = case.data.geometry
    layouts = case.data.layouts
    if field_layout_kind == "regular" and material_wall:
        wall_paths = materialize_microcase("C07", directory / "wall")
        geometry = load_case(wall_paths.case_path).data.geometry
    elif field_layout_kind in {"p1", "q1"}:
        geometry, layout = _unstructured_harmonic_geometry(
            field_layout_kind,
            material_wall=material_wall,
        )
        layouts = (layout,)
    elif field_layout_kind != "regular":
        raise ValueError(f"unknown harmonic field layout kind: {field_layout_kind}")
    original = case.data.sources[0]

    def repeated(values: np.ndarray) -> np.ndarray:
        return np.repeat(values[:1], particle_count, axis=0)

    source = replace(
        original,
        particle_id=np.arange(401, 401 + particle_count, dtype="<i8"),
        release_time_s=repeated(original.release_time_s),
        position_m=np.repeat(
            initial_position_m[None, :].astype("<f8"),
            particle_count,
            axis=0,
        ),
        velocity_m_s=np.repeat(
            initial_velocity_m_s[None, :].astype("<f8"),
            particle_count,
            axis=0,
        ),
        charge_number=repeated(original.charge_number),
        mass_kg=repeated(original.mass_kg),
        drag_diameter_m=repeated(original.drag_diameter_m),
        electrostatic_radius_m=repeated(original.electrostatic_radius_m),
        displaced_volume_m3=repeated(original.displaced_volume_m3),
        model_weight=repeated(original.model_weight),
        material_id=repeated(original.material_id),
    )
    charge_to_acceleration = (
        float(source.charge_number[0]) * 1.602176634e-19 / float(source.mass_kg[0])
    )
    node_x_m = (
        np.asarray([-1.0, -1.0, 1.0, 1.0], dtype="<f8")
        if field_layout_kind == "regular"
        else geometry.nodes_m[:, 0]
    )
    acceleration_x_m_s2 = -(angular_frequency_s_inv**2) * node_x_m
    electric_x_v_m = acceleration_x_m_s2 / charge_to_acceleration
    electric_values = np.column_stack((electric_x_v_m, np.zeros(electric_x_v_m.size, dtype="<f8")))
    fields = tuple(
        replace(
            field,
            layout=field.layout if field_layout_kind == "regular" else "unstructured",
            values=electric_values,
        )
        if field.name == "electric_field"
        else field
        for field in case.data.fields
    )
    data_path = paths.case_path.with_name("harmonic-electric.h5")
    info = write(
        data_path,
        replace(
            case.data,
            geometry=geometry,
            layouts=layouts,
            fields=fields,
            sources=(source,),
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"] = {"start_s": 0.0, "end_s": end_time_s, "dt_s": step_s}
    if frame_times is None:
        document["output"]["trajectories"] = None
    else:
        document["output"]["trajectories"]["schedule"] = {"explicit_times_s": frame_times}
    if material_wall:
        document["boundaries"] = [{"boundary_group": "wall", "priority": 10, "law": "stick"}]
    case_path = paths.case_path.with_name("harmonic-electric.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _loose_global_bound_harmonic_wall_case(directory: Path) -> Path:
    """Keep the traversed cell harmonic while making the run-global bound huge."""

    case_path = _harmonic_electric_case(
        directory,
        step_s=0.5,
        end_time_s=0.5,
        angular_frequency_s_inv=1.0e-6,
        initial_position_m=np.asarray([0.5, 0.5]),
        initial_velocity_m_s=np.asarray([2.0, 0.0]),
        frame_times=None,
        material_wall=True,
    )
    case = load_case(case_path)
    layout = case.data.layouts[0]
    if not isinstance(layout, RegularLayout):
        raise TypeError("loose-bound harmonic case requires a regular field layout")
    expanded_layout = replace(
        layout,
        axis0_m=np.asarray([-1.0e3, 0.0, 1.0], dtype="<f8"),
        cell_support=np.ones((2, 1), dtype="<u1"),
    )
    source = case.data.sources[0]
    charge_to_acceleration = (
        float(source.charge_number[0]) * ELEMENTARY_CHARGE_C / float(source.mass_kg[0])
    )
    acceleration_x_m_s2 = np.repeat(np.asarray([100.0, 0.0, -1.0e-12]), 2)
    electric_values = np.column_stack(
        (
            acceleration_x_m_s2 / charge_to_acceleration,
            np.zeros(acceleration_x_m_s2.size, dtype="<f8"),
        )
    )
    fields = tuple(
        replace(field, values=electric_values) if field.name == "electric_field" else field
        for field in case.data.fields
    )
    nodes_m = case.data.geometry.nodes_m.copy()
    nodes_m[[0, 3], 0] = -1.0e3
    geometry = replace(case.data.geometry, nodes_m=nodes_m)
    data_path = case_path.with_name("loose-bound-harmonic-electric.h5")
    info = write(
        data_path,
        replace(case.data, geometry=geometry, layouts=(expanded_layout,), fields=fields),
    )
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["solver"]["event"]["max_refinements"] = 48
    loose_case_path = case_path.with_name("loose-bound-harmonic-electric.yaml")
    loose_case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return loose_case_path


def _loose_global_bound_harmonic_interior_case(directory: Path) -> Path:
    """Keep the dense path interior while broadening only its global safety bound."""

    case_path = _harmonic_electric_case(
        directory,
        step_s=0.5,
        end_time_s=0.5,
        angular_frequency_s_inv=1.0e-6,
        initial_position_m=np.asarray([0.5, 0.5]),
        initial_velocity_m_s=np.asarray([0.1, 0.0]),
        frame_times=None,
        material_wall=True,
    )
    case = load_case(case_path)
    layout = case.data.layouts[0]
    if not isinstance(layout, RegularLayout):
        raise TypeError("loose-bound harmonic case requires a regular field layout")
    expanded_layout = replace(
        layout,
        axis0_m=np.asarray([-1.0e3, 0.0, 1.0e3], dtype="<f8"),
        cell_support=np.ones((2, 1), dtype="<u1"),
    )
    source = case.data.sources[0]
    charge_to_acceleration = (
        float(source.charge_number[0]) * ELEMENTARY_CHARGE_C / float(source.mass_kg[0])
    )
    acceleration_x_m_s2 = np.repeat(np.asarray([100.0, 0.0, -1.0e-12]), 2)
    electric_values = np.column_stack(
        (
            acceleration_x_m_s2 / charge_to_acceleration,
            np.zeros(acceleration_x_m_s2.size, dtype="<f8"),
        )
    )
    fields = tuple(
        replace(field, values=electric_values) if field.name == "electric_field" else field
        for field in case.data.fields
    )
    data_path = case_path.with_name("loose-interior-harmonic-electric.h5")
    info = write(
        data_path,
        replace(case.data, layouts=(expanded_layout,), fields=fields),
    )
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    loose_case_path = case_path.with_name("loose-interior-harmonic-electric.yaml")
    loose_case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return loose_case_path


def _unstructured_harmonic_geometry(
    kind: str,
    *,
    material_wall: bool,
) -> tuple[GeometryData, P1TriLayout | Q1QuadLayout]:
    nodes = np.asarray(
        (
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            if kind == "p1"
            else [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.2, 1.0]]
        ),
        dtype="<f8",
    )
    if kind == "p1":
        connectivity = np.asarray([[0, 1, 2], [0, 2, 3]], dtype="<i8")
        owner_cell_type = np.ones(4, dtype="<u1")
        owner_cell_local_index = np.asarray([0, 0, 1, 1], dtype="<i8")
        layout: P1TriLayout | Q1QuadLayout = P1TriLayout(
            "unstructured",
            nodes.copy(),
            connectivity.copy(),
            np.ones(2, dtype="<u1"),
        )
        volume = {
            "tri3": connectivity,
            "tri3_domain_id": np.zeros(2, dtype="<i4"),
        }
    else:
        connectivity = np.asarray([[0, 1, 2, 3]], dtype="<i8")
        owner_cell_type = np.full(4, 2, dtype="<u1")
        owner_cell_local_index = np.zeros(4, dtype="<i8")
        layout = Q1QuadLayout(
            "unstructured",
            nodes.copy(),
            connectivity.copy(),
            np.ones(1, dtype="<u1"),
        )
        volume = {
            "quad4": connectivity,
            "quad4_domain_id": np.zeros(1, dtype="<i4"),
        }
    if material_wall:
        boundary = BoundaryData(
            line2=np.asarray([[0, 1], [1, 2], [2, 3], [3, 0]], dtype="<i8"),
            boundary_id=np.full(4, 10, dtype="<i4"),
            group_id=np.zeros(4, dtype="<i4"),
            material_id=np.zeros(4, dtype="<i4"),
            owner_cell_type=owner_cell_type,
            owner_cell_local_index=owner_cell_local_index,
            orientation=np.ones(4, dtype="<i1"),
        )
        group_names = ("wall",)
    else:
        boundary = BoundaryData(
            line2=np.empty((0, 2), dtype="<i8"),
            boundary_id=np.empty(0, dtype="<i4"),
            group_id=np.empty(0, dtype="<i4"),
            material_id=np.empty(0, dtype="<i4"),
            owner_cell_type=np.empty(0, dtype="<u1"),
            owner_cell_local_index=np.empty(0, dtype="<i8"),
            orientation=np.empty(0, dtype="<i1"),
        )
        group_names = ()
    return (
        GeometryData(
            nodes_m=nodes,
            boundary=boundary,
            group_names=group_names,
            **volume,
        ),
        layout,
    )


@pytest.mark.parametrize(
    ("integrator", "minimum_order"),
    [("rk4_fixed", 3.5), ("exponential_midpoint", 1.8)],
)
def test_barnes_ion_drag_constant_field_time_convergence(
    tmp_path: Path,
    integrator: str,
    minimum_order: float,
) -> None:
    end_time = 0.5
    reference = _barnes_reference_state(end_time, step_count=32768)
    position_error: list[float] = []
    velocity_error: list[float] = []
    last_result = None
    for index, dt_s in enumerate((0.125, 0.0625, 0.03125)):
        case_path = _barnes_ion_drag_case(
            tmp_path / f"barnes-{integrator}-{index}",
            integrator=integrator,
            dt_s=dt_s,
            end_time_s=end_time,
        )
        output = tmp_path / f"barnes-result-{integrator}-{index}"
        simulate(load_case(case_path), output)
        last_result = open_result(output)
        final = last_result.read_final()
        position_error.append(float(np.linalg.norm(final.position_m[0] - reference[:2])))
        velocity_error.append(float(np.linalg.norm(final.velocity_m_s[0] - reference[2:])))

    assert last_result is not None
    for errors in (position_error, velocity_error):
        orders = [math.log2(errors[index] / errors[index + 1]) for index in (0, 1)]
        assert min(orders) >= minimum_order
    assert last_result.manifest["resolved"]["physics_models"]["ion_drag"] == {
        "model": "barnes_collisionless",
        "revision": _BARNES_ION_DRAG_REVISION,
    }
    assert last_result.manifest["compiled_cpu_tile_revision"] == "compiled_cpu_tile_v18"
    assert last_result.manifest["physics_catalog_revision"] == "inertial_langevin_rz_catalog_v17"
    assert (
        last_result.manifest["physics_runtime_revision"]
        == "signed_ion_compiled_physics_runtime_v20"
    )


def test_barnes_ion_drag_xy_and_rz_use_the_same_stage_physics(tmp_path: Path) -> None:
    outputs = []
    for axisymmetric in (False, True):
        case_path = _barnes_ion_drag_case(
            tmp_path / f"barnes-basis-{axisymmetric}",
            integrator="rk4_fixed",
            dt_s=0.03125,
            end_time_s=0.5,
            axisymmetric_rz=axisymmetric,
        )
        output = tmp_path / f"barnes-basis-result-{axisymmetric}"
        simulate(load_case(case_path), output)
        outputs.append(open_result(output).read_final())

    shift = np.asarray([1.5, 0.0])
    np.testing.assert_allclose(
        outputs[0].position_m,
        outputs[1].position_m - shift,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        outputs[0].velocity_m_s,
        outputs[1].velocity_m_s,
        rtol=0.0,
        atol=2.0e-16,
    )
    np.testing.assert_array_equal(outputs[0].charge_number, outputs[1].charge_number)


def _barnes_ion_drag_case(
    directory: Path,
    *,
    integrator: str,
    dt_s: float,
    end_time_s: float,
    axisymmetric_rz: bool = False,
) -> Path:
    paths = materialize_microcase("C07", directory)
    case = load_case(paths.case_path)
    radial_shift = np.asarray([1.5, 0.0]) if axisymmetric_rz else np.zeros(2)
    source = replace(
        case.data.sources[0],
        position_m=case.data.sources[0].position_m + radial_shift,
        velocity_m_s=np.asarray([[0.1, 0.0]]),
        charge_number=np.asarray([-100.0]),
        mass_kg=np.asarray([4.0e-17]),
    )
    vector_components = ("r", "z") if axisymmetric_rz else ("x", "y")
    vector_basis = "axisymmetric_rz" if axisymmetric_rz else "cartesian_xy"
    layout = RegularLayout(
        "plasma",
        np.asarray([radial_shift[0], radial_shift[0] + 1.0]),
        np.asarray([0.0, 1.0]),
        np.ones((1, 1), dtype=np.uint8),
    )

    def field(
        name: str,
        value: tuple[float, ...],
        components: tuple[str, ...],
        basis: str,
        unit: str,
    ) -> FieldData:
        values = np.repeat(np.asarray([value]), 4, axis=0)
        return FieldData(name, "plasma", "node", components, basis, values, unit)

    fields = (
        field("ne", (3.0e17,), ("value",), "scalar", "1/m^3"),
        field("ni", (3.0e17,), ("value",), "scalar", "1/m^3"),
        field("te", (1.0e5,), ("value",), "scalar", "K"),
        field("ti", (1.0e4,), ("value",), "scalar", "K"),
        field("ui", (0.8, -0.1), vector_components, vector_basis, "m/s"),
        field("ion_mfp", (1.0,), ("value",), "scalar", "m"),
    )
    boundaryless = BoundaryData(
        line2=np.empty((0, 2), dtype=np.int64),
        boundary_id=np.empty(0, dtype=np.int32),
        group_id=np.empty(0, dtype=np.int32),
        material_id=np.empty(0, dtype=np.int32),
        owner_cell_type=np.empty(0, dtype=np.uint8),
        owner_cell_local_index=np.empty(0, dtype=np.int64),
        orientation=np.empty(0, dtype=np.int8),
    )
    data_path = paths.case_path.with_name("barnes-ion-drag.h5")
    info = write(
        data_path,
        replace(
            case.data,
            coordinate_system=("axisymmetric_rz" if axisymmetric_rz else "cartesian_xy"),
            geometry=replace(
                case.data.geometry,
                nodes_m=case.data.geometry.nodes_m + radial_shift,
                boundary=boundaryless,
                group_names=(),
            ),
            layouts=(layout,),
            fields=fields,
            sources=(source,),
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["motion"]["mode"] = "axisymmetric_rz_meridional" if axisymmetric_rz else "cartesian_xy"
    document["time"] = {"start_s": 0.0, "end_s": end_time_s, "dt_s": dt_s}
    document["solver"]["integrator"] = integrator
    document["physics"] = {
        "charge": {"model": "fixed"},
        "ion_drag": {
            "model": "barnes_collisionless",
            "revision": _BARNES_ION_DRAG_REVISION,
            "electron_number_density_field": "ne",
            "positive_ion_number_density_field": "ni",
            "electron_temperature_field": "te",
            "positive_ion_temperature_field": "ti",
            "positive_ion_velocity_field": "ui",
            "ion_neutral_mean_free_path_field": "ion_mfp",
            "positive_ion_mass_kg": _CONTINUOUS_CHARGE_ION_MASS_KG,
            "maximum_ion_drift_ratio": 1.0e-3,
            "applicability": "error",
        },
    }
    document["boundaries"] = []
    document["output"] = {"trajectories": None}
    case_path = paths.case_path.with_name("barnes-ion-drag.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _barnes_reference_state(end_time_s: float, *, step_count: int) -> np.ndarray:
    state = np.asarray([0.25, 0.25, 0.1, 0.0], dtype=np.float64)
    step = end_time_s / step_count
    for _ in range(step_count):
        first = _barnes_reference_rhs(state)
        second = _barnes_reference_rhs(state + 0.5 * step * first)
        third = _barnes_reference_rhs(state + 0.5 * step * second)
        fourth = _barnes_reference_rhs(state + step * third)
        state += step * (first + 2.0 * second + 2.0 * third + fourth) / 6.0
    return state


def _barnes_reference_rhs(state: np.ndarray) -> np.ndarray:
    electron_density = 3.0e17
    ion_density = 3.0e17
    electron_temperature = 1.0e5
    ion_temperature = 1.0e4
    ion_mass = _CONTINUOUS_CHARGE_ION_MASS_KG
    particle_mass = 4.0e-17
    radius = 1.0e-6
    charge_number = -100.0
    relative = np.asarray([0.8, -0.1]) - state[2:]
    relative_speed_square = float(np.dot(relative, relative))
    thermal_speed_square = 8.0 * BOLTZMANN_J_K * ion_temperature / (math.pi * ion_mass)
    effective_speed_square = relative_speed_square + thermal_speed_square
    effective_speed = math.sqrt(effective_speed_square)
    debye_length = math.sqrt(
        VACUUM_PERMITTIVITY_F_M
        * BOLTZMANN_J_K
        / (
            ELEMENTARY_CHARGE_C**2
            * (electron_density / electron_temperature + ion_density / ion_temperature)
        )
    )
    capacitance = 4.0 * math.pi * VACUUM_PERMITTIVITY_F_M * radius * (1.0 + radius / debye_length)
    potential = charge_number * ELEMENTARY_CHARGE_C / capacitance
    orbital = (
        abs(charge_number)
        * ELEMENTARY_CHARGE_C**2
        / (4.0 * math.pi * VACUUM_PERMITTIVITY_F_M * ion_mass * effective_speed_square)
    )
    collection_square = radius**2 * (
        1.0 - 2.0 * ELEMENTARY_CHARGE_C * potential / (ion_mass * effective_speed_square)
    )
    coulomb_logarithm = 0.5 * math.log(
        (debye_length**2 + orbital**2) / (collection_square + orbital**2)
    )
    cross_section = math.pi * collection_square + 4.0 * math.pi * orbital**2 * coulomb_logarithm
    acceleration = ion_density * ion_mass * effective_speed * cross_section * relative
    acceleration /= particle_mass
    return np.asarray([state[2], state[3], acceleration[0], acceleration[1]])
