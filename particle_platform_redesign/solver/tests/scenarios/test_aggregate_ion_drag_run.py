from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import BoundaryData, FieldData, RegularLayout, write
from tests.verification.microcases import materialize_microcase

_RELATIVE_FLOW_REVISION = "relative_flow_screened_collection_orbital_aggregate_ion_v1"
_IMAGE_REVISION = "electric_field_directed_image_orbital_sensitivity_v1"
_PARTICLE_MASS_KG = 2.0e-18
_PARTICLE_RADIUS_M = 5.0e-8
_CHARGE_NUMBER = -180.0
_ION_DENSITY_M3 = 8.0e14
_ELECTRON_THERMAL_VOLTAGE_V = 3.5
_ION_THERMAL_VOLTAGE_V = 0.03
_ION_VELOCITY_M_S = np.asarray([20.0, 4.0])
_ION_MASS_KG = 7.0e-26
_SCREENING_LENGTH_M = 2.0e-5
_ION_MEAN_FREE_PATH_M = 7.0e-6
_ELECTRIC_FIELD_V_M = np.asarray([40.0, -30.0])
_DOMAIN_WIDTH_M = 20.0


@pytest.mark.parametrize(
    ("integrator", "minimum_order", "path_kind"),
    [
        ("rk4_fixed", 3.5, "rk4_dense"),
        ("exponential_midpoint", 1.8, "exponential_midpoint_reintegrated"),
    ],
)
def test_relative_flow_public_run_converges_in_xy(
    tmp_path: Path,
    integrator: Literal["rk4_fixed", "exponential_midpoint"],
    minimum_order: float,
    path_kind: str,
) -> None:
    states: list[np.ndarray] = []
    last_result = None
    end_time_s = 0.5
    for index, dt_s in enumerate((0.03125, 0.015625, 0.0078125, 0.00390625)):
        case_path = _case(
            tmp_path / f"relative-flow-{index}",
            model="relative_flow",
            coordinate_system="cartesian_xy",
            integrator=integrator,
            dt_s=dt_s,
            end_time_s=end_time_s,
        )
        output = tmp_path / f"relative-flow-result-{index}"

        simulate(load_case(case_path), output)
        last_result = open_result(output)
        final = last_result.read_final()
        np.testing.assert_array_equal(final.lifecycle, [1])
        states.append(np.concatenate((final.position_m[0], final.velocity_m_s[0])))

    assert last_result is not None
    differences = np.asarray(
        [np.linalg.norm(states[index] - states[index + 1]) for index in range(3)]
    )
    assert bool((differences > 0.0).all())
    observed_orders = np.log2(differences[:-1] / differences[1:])
    assert float(np.min(observed_orders)) >= minimum_order

    final = last_result.read_final()
    position_change = final.position_m[0] - _initial_position("cartesian_xy")
    velocity = final.velocity_m_s[0]
    orthogonal = np.asarray([-_ION_VELOCITY_M_S[1], _ION_VELOCITY_M_S[0]])
    assert float(np.dot(velocity, _ION_VELOCITY_M_S)) > 0.0
    assert np.linalg.norm(velocity) < np.linalg.norm(_ION_VELOCITY_M_S)
    np.testing.assert_allclose(np.dot(velocity, orthogonal), 0.0, rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(
        np.dot(position_change, orthogonal),
        0.0,
        rtol=0.0,
        atol=2.0e-13,
    )
    assert last_result.manifest["resolved"]["physics_models"]["ion_drag"] == {
        "model": "screened_collection_orbital",
        "revision": _RELATIVE_FLOW_REVISION,
    }
    assert last_result.manifest["resolved"]["path_kind"] == path_kind
    assert last_result.manifest["data_coordinate_system"] == "cartesian_xy"


@pytest.mark.parametrize(
    ("integrator", "path_kind"),
    [
        ("rk4_fixed", "rk4_dense"),
        ("exponential_midpoint", "exponential_midpoint_reintegrated"),
    ],
)
def test_image_public_run_is_constant_acceleration_in_rz(
    tmp_path: Path,
    integrator: Literal["rk4_fixed", "exponential_midpoint"],
    path_kind: str,
) -> None:
    end_time_s = 0.1
    case_path = _case(
        tmp_path / "image",
        model="image",
        coordinate_system="axisymmetric_rz",
        integrator=integrator,
        dt_s=0.025,
        end_time_s=end_time_s,
    )
    output = tmp_path / "image-result"

    simulate(load_case(case_path), output)
    result = open_result(output)
    final = result.read_final()
    frames = list(result.iter_frames())

    initial_position = _initial_position("axisymmetric_rz")
    velocity = final.velocity_m_s[0]
    position_change = final.position_m[0] - initial_position
    assert np.linalg.norm(velocity) > 1.0
    assert float(np.dot(velocity, _ELECTRIC_FIELD_V_M)) > 0.0
    np.testing.assert_allclose(
        velocity[0] * _ELECTRIC_FIELD_V_M[1],
        velocity[1] * _ELECTRIC_FIELD_V_M[0],
        rtol=0.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        position_change,
        0.5 * velocity * end_time_s,
        rtol=2.0e-13,
        atol=2.0e-13,
    )
    assert [frame.time_s for frame in frames] == [0.0, end_time_s]
    np.testing.assert_allclose(frames[0].position_m[0], initial_position, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(frames[-1].position_m, final.position_m, rtol=0.0, atol=0.0)
    assert result.manifest["resolved"]["physics_models"]["ion_drag"] == {
        "model": "image_orbital_sensitivity",
        "revision": _IMAGE_REVISION,
    }
    assert result.manifest["resolved"]["path_kind"] == path_kind
    assert result.manifest["data_coordinate_system"] == "axisymmetric_rz"


@pytest.mark.parametrize(
    ("integrator", "path_kind"),
    [
        ("rk4_fixed", "rk4_dense"),
        ("exponential_midpoint", "exponential_midpoint_reintegrated"),
    ],
)
def test_relative_flow_public_run_uses_the_shared_material_event_path(
    tmp_path: Path,
    integrator: Literal["rk4_fixed", "exponential_midpoint"],
    path_kind: str,
) -> None:
    case_path = _case(
        tmp_path / f"relative-flow-wall-{integrator}",
        model="relative_flow",
        coordinate_system="cartesian_xy",
        integrator=integrator,
        dt_s=0.125,
        end_time_s=2.0,
        material_boundaries=True,
    )
    output = tmp_path / f"relative-flow-wall-result-{integrator}"
    simulate(load_case(case_path), output)
    result = open_result(output)
    event = result.read_boundary_events()

    np.testing.assert_array_equal(event.particle_id, [701])
    np.testing.assert_allclose(event.position_m, [[_DOMAIN_WIDTH_M, 12.0]], atol=2.0e-11)
    np.testing.assert_array_equal(event.normal, [[1.0, 0.0]])
    np.testing.assert_array_equal(event.velocity_post_m_s, [[0.0, 0.0]])
    np.testing.assert_array_equal(event.law_id, ["stick"])
    np.testing.assert_array_equal(event.outcome, ["stuck"])
    np.testing.assert_array_equal(result.read_final().lifecycle, [2])
    assert result.manifest["resolved"]["path_kind"] == path_kind


def _case(
    directory: Path,
    *,
    model: Literal["relative_flow", "image"],
    coordinate_system: Literal["cartesian_xy", "axisymmetric_rz"],
    integrator: Literal["rk4_fixed", "exponential_midpoint"],
    dt_s: float,
    end_time_s: float,
    material_boundaries: bool = False,
) -> Path:
    paths = materialize_microcase("C07", directory)
    case = load_case(paths.case_path)
    radial_shift = (
        np.asarray([_DOMAIN_WIDTH_M, 0.0])
        if coordinate_system == "axisymmetric_rz"
        else np.zeros(2)
    )
    vector_components = ("r", "z") if coordinate_system == "axisymmetric_rz" else ("x", "y")
    vector_basis = "axisymmetric_rz" if coordinate_system == "axisymmetric_rz" else "cartesian_xy"
    layout = RegularLayout(
        "plasma",
        np.asarray([radial_shift[0], radial_shift[0] + _DOMAIN_WIDTH_M]),
        np.asarray([0.0, _DOMAIN_WIDTH_M]),
        np.ones((1, 1), dtype=np.uint8),
    )

    def constant_field(
        name: str,
        value: tuple[float, ...],
        components: tuple[str, ...],
        basis: str,
        unit: str,
    ) -> FieldData:
        values = np.repeat(np.asarray([value], dtype="<f8"), 4, axis=0)
        return FieldData(name, "plasma", "node", components, basis, values, unit)

    fields = (
        constant_field("ni", (_ION_DENSITY_M3,), ("value",), "scalar", "1/m^3"),
        constant_field(
            "te_v",
            (_ELECTRON_THERMAL_VOLTAGE_V,),
            ("value",),
            "scalar",
            "V",
        ),
        constant_field(
            "ti_v",
            (_ION_THERMAL_VOLTAGE_V,),
            ("value",),
            "scalar",
            "V",
        ),
        constant_field(
            "ui",
            tuple(_ION_VELOCITY_M_S),
            vector_components,
            vector_basis,
            "m/s",
        ),
        constant_field("mi", (_ION_MASS_KG,), ("value",), "scalar", "kg"),
        constant_field(
            "screening",
            (_SCREENING_LENGTH_M,),
            ("value",),
            "scalar",
            "m",
        ),
        constant_field(
            "ion_mfp",
            (_ION_MEAN_FREE_PATH_M,),
            ("value",),
            "scalar",
            "m",
        ),
        constant_field(
            "efield",
            tuple(_ELECTRIC_FIELD_V_M),
            vector_components,
            vector_basis,
            "V/m",
        ),
    )
    source = replace(
        case.data.sources[0],
        position_m=_initial_position(coordinate_system)[None, :],
        velocity_m_s=np.zeros((1, 2)),
        charge_number=np.asarray([_CHARGE_NUMBER]),
        mass_kg=np.asarray([_PARTICLE_MASS_KG]),
        drag_diameter_m=np.asarray([2.0 * _PARTICLE_RADIUS_M]),
        electrostatic_radius_m=np.asarray([_PARTICLE_RADIUS_M]),
        displaced_volume_m3=np.asarray([0.0]),
    )
    boundary = case.data.geometry.boundary
    group_names = case.data.geometry.group_names
    if not material_boundaries:
        boundary = BoundaryData(
            line2=np.empty((0, 2), dtype=np.int64),
            boundary_id=np.empty(0, dtype=np.int32),
            group_id=np.empty(0, dtype=np.int32),
            material_id=np.empty(0, dtype=np.int32),
            owner_cell_type=np.empty(0, dtype=np.uint8),
            owner_cell_local_index=np.empty(0, dtype=np.int64),
            orientation=np.empty(0, dtype=np.int8),
        )
        group_names = ()
    data_path = paths.case_path.with_name("aggregate-ion-drag.h5")
    info = write(
        data_path,
        replace(
            case.data,
            coordinate_system=coordinate_system,
            geometry=replace(
                case.data.geometry,
                nodes_m=case.data.geometry.nodes_m * _DOMAIN_WIDTH_M + radial_shift,
                boundary=boundary,
                group_names=group_names,
            ),
            layouts=(layout,),
            fields=fields,
            sources=(source,),
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["motion"]["mode"] = (
        "axisymmetric_rz_meridional" if coordinate_system == "axisymmetric_rz" else "cartesian_xy"
    )
    document["time"] = {"start_s": 0.0, "end_s": end_time_s, "dt_s": dt_s}
    document["solver"]["integrator"] = integrator
    document["physics"] = {
        "charge": {"model": "fixed"},
        "ion_drag": _ion_drag_model(model),
    }
    if not material_boundaries:
        document["boundaries"] = []
    document["output"] = {
        "trajectories": {
            "selection": "all",
            "schedule": {"explicit_times_s": [0.0, end_time_s]},
        }
    }
    case_path = paths.case_path.with_name("aggregate-ion-drag.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _ion_drag_model(model: Literal["relative_flow", "image"]) -> dict[str, object]:
    common: dict[str, object] = {
        "positive_ion_number_density_field": "ni",
        "positive_ion_thermal_voltage_field": "ti_v",
        "positive_ion_velocity_field": "ui",
        "effective_positive_ion_mass_field": "mi",
        "screening_length_field": "screening",
        "applicability": "error",
    }
    if model == "relative_flow":
        return {
            "model": "screened_collection_orbital",
            "revision": _RELATIVE_FLOW_REVISION,
            **common,
            "ion_neutral_mean_free_path_field": "ion_mfp",
            "maximum_relative_ion_speed_m_s": 500.0,
        }
    return {
        "model": "image_orbital_sensitivity",
        "revision": _IMAGE_REVISION,
        **common,
        "electron_thermal_voltage_field": "te_v",
        "electric_field": "efield",
    }


def _initial_position(
    coordinate_system: Literal["cartesian_xy", "axisymmetric_rz"],
) -> np.ndarray:
    radial_shift = _DOMAIN_WIDTH_M if coordinate_system == "axisymmetric_rz" else 0.0
    return np.asarray([radial_shift + 0.5 * _DOMAIN_WIDTH_M, 0.5 * _DOMAIN_WIDTH_M])
