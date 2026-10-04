from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import FieldData, write
from tests.verification.microcases import materialize_microcase

_REVISION = "quasistatic_spherical_gradient_e2_v1"
_EPSILON_0_F_M = 8.8541878128e-12
_PARTICLE_MASS_KG = 1.0e-15
_PARTICLE_RADIUS_M = 1.0e-6
_RELATIVE_PERMITTIVITY = 2.0
_CM_FACTOR = -0.25
_ANGULAR_FREQUENCY_S_INV = 2.0
_INITIAL_LOCAL_POSITION_M = np.asarray([0.3, -0.2])
_INITIAL_VELOCITY_M_S = np.asarray([0.0, 0.4])
_END_TIME_S = 0.75


@pytest.mark.parametrize(
    ("integrator", "coordinate_system", "minimum_order", "path_kind"),
    [
        ("rk4_fixed", "cartesian_xy", 3.5, "rk4_dense"),
        (
            "exponential_midpoint",
            "axisymmetric_rz",
            1.8,
            "exponential_midpoint_reintegrated",
        ),
    ],
)
def test_dep_quadratic_potential_converges_with_pairwise_basis_coverage(
    tmp_path: Path,
    integrator: Literal["rk4_fixed", "exponential_midpoint"],
    coordinate_system: Literal["cartesian_xy", "axisymmetric_rz"],
    minimum_order: float,
    path_kind: str,
) -> None:
    expected = _closed_form_state(_END_TIME_S)
    errors: list[float] = []
    last_result = None
    radial_shift_m = 2.0 if coordinate_system == "axisymmetric_rz" else 0.0

    for index, dt_s in enumerate((0.125, 0.0625, 0.03125)):
        case_path = _dep_case(
            tmp_path / f"{integrator}-{coordinate_system}-{index}",
            integrator=integrator,
            coordinate_system=coordinate_system,
            dt_s=dt_s,
        )
        output = tmp_path / f"dep-result-{integrator}-{coordinate_system}-{index}"
        simulate(load_case(case_path), output)
        last_result = open_result(output)
        final = last_result.read_final()
        np.testing.assert_array_equal(final.lifecycle, [1])
        local_position = final.position_m[0] - np.asarray([radial_shift_m, 0.0])
        state = np.concatenate((local_position, final.velocity_m_s[0]))
        errors.append(float(np.linalg.norm(state - expected)))

    assert last_result is not None
    assert errors[0] > errors[1] > errors[2] > 0.0
    observed_orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert float(np.min(observed_orders)) >= minimum_order
    assert last_result.manifest["resolved"]["physics_models"]["dielectrophoresis"] == {
        "model": "quasistatic_spherical",
        "revision": _REVISION,
    }
    assert last_result.manifest["resolved"]["path_kind"] == path_kind
    assert last_result.manifest["data_coordinate_system"] == coordinate_system


def _dep_case(
    directory: Path,
    *,
    integrator: Literal["rk4_fixed", "exponential_midpoint"],
    coordinate_system: Literal["cartesian_xy", "axisymmetric_rz"],
    dt_s: float,
) -> Path:
    paths = materialize_microcase("C04", directory)
    case = load_case(paths.case_path)
    radial_shift_m = 2.0 if coordinate_system == "axisymmetric_rz" else 0.0
    shift = np.asarray([radial_shift_m, 0.0])
    components = ("r", "z") if coordinate_system == "axisymmetric_rz" else ("x", "y")
    basis = "axisymmetric_rz" if coordinate_system == "axisymmetric_rz" else "cartesian_xy"
    original_layout = case.data.layouts[0]
    layout = replace(
        original_layout,
        axis0_m=original_layout.axis0_m + radial_shift_m,
    )

    dep_factor_m4_v2_s2 = (
        2.0
        * math.pi
        * _EPSILON_0_F_M
        * _RELATIVE_PERMITTIVITY
        * _CM_FACTOR
        * _PARTICLE_RADIUS_M**3
        / _PARTICLE_MASS_KG
    )
    gradient_slope_V2_m4 = -(_ANGULAR_FREQUENCY_S_INV**2) / dep_factor_m4_v2_s2
    local_axis0_m = layout.axis0_m - radial_shift_m
    node_local_axis0_m = np.repeat(local_axis0_m, layout.axis1_m.size)
    gradient_values = np.column_stack(
        (
            gradient_slope_V2_m4 * node_local_axis0_m,
            np.zeros(node_local_axis0_m.size),
        )
    )
    gradient_field = FieldData(
        "gradient_mean_e_squared",
        layout.name,
        "node",
        components,
        basis,
        gradient_values,
        "V^2/m^3",
    )
    original_source = case.data.sources[0]
    source = replace(
        original_source,
        particle_id=original_source.particle_id[:1],
        release_time_s=np.asarray([0.0]),
        position_m=(_INITIAL_LOCAL_POSITION_M + shift)[None, :],
        velocity_m_s=_INITIAL_VELOCITY_M_S[None, :],
        charge_number=np.asarray([0.0]),
        mass_kg=np.asarray([_PARTICLE_MASS_KG]),
        drag_diameter_m=np.asarray([2.0 * _PARTICLE_RADIUS_M]),
        electrostatic_radius_m=np.asarray([_PARTICLE_RADIUS_M]),
        displaced_volume_m3=np.asarray([0.0]),
        model_weight=original_source.model_weight[:1],
        material_id=original_source.material_id[:1],
    )
    data_path = paths.case_path.with_name("dielectrophoresis.h5")
    info = write(
        data_path,
        replace(
            case.data,
            coordinate_system=coordinate_system,
            geometry=replace(
                case.data.geometry,
                nodes_m=case.data.geometry.nodes_m + shift,
            ),
            layouts=(layout,),
            fields=(gradient_field,),
            sources=(source,),
        ),
    )

    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["motion"]["mode"] = (
        "axisymmetric_rz_meridional" if coordinate_system == "axisymmetric_rz" else "cartesian_xy"
    )
    document["time"] = {"start_s": 0.0, "end_s": _END_TIME_S, "dt_s": dt_s}
    document["solver"]["integrator"] = integrator
    document["physics"] = {
        "charge": {"model": "fixed"},
        "dielectrophoresis": {
            "model": "quasistatic_spherical",
            "revision": _REVISION,
            "gradient_mean_e_squared_field": "gradient_mean_e_squared",
            "medium_relative_permittivity": _RELATIVE_PERMITTIVITY,
            "real_clausius_mossotti_factor": _CM_FACTOR,
            "maximum_point_dipole_radius_m": _PARTICLE_RADIUS_M,
        },
    }
    document["output"] = {"trajectories": None}
    case_path = paths.case_path.with_name("dielectrophoresis.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _closed_form_state(time_s: float) -> np.ndarray:
    phase = _ANGULAR_FREQUENCY_S_INV * time_s
    position = np.asarray(
        [
            _INITIAL_LOCAL_POSITION_M[0] * math.cos(phase)
            + _INITIAL_VELOCITY_M_S[0] * math.sin(phase) / _ANGULAR_FREQUENCY_S_INV,
            _INITIAL_LOCAL_POSITION_M[1] + _INITIAL_VELOCITY_M_S[1] * time_s,
        ]
    )
    velocity = np.asarray(
        [
            -_INITIAL_LOCAL_POSITION_M[0] * _ANGULAR_FREQUENCY_S_INV * math.sin(phase)
            + _INITIAL_VELOCITY_M_S[0] * math.cos(phase),
            _INITIAL_VELOCITY_M_S[1],
        ]
    )
    return np.concatenate((position, velocity))
