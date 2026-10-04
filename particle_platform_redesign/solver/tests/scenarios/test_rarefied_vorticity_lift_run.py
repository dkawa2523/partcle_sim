from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import FieldData, RegularLayout, write
from tests.verification.microcases import materialize_microcase

_REVISION = "rarefied_vorticity_sensitivity_rz_v1"
_LIFT_COEFFICIENT = 1.0
_GAS_DENSITY_KG_M3 = 1.0e-2
_GAS_MEAN_FREE_PATH_M = 1.0e-3
_PARTICLE_RADIUS_M = 1.0e-5
_PARTICLE_DIAMETER_M = 2.0 * _PARTICLE_RADIUS_M
_PARTICLE_MASS_KG = 2.0 * math.pi * 1.0e-15
_SHEAR_RATE_S_INV = 2.0
_INITIAL_POSITION_M = np.asarray([2.25, 0.20])
_INITIAL_VELOCITY_M_S = np.asarray([0.05, 0.10])
_END_TIME_S = 0.75


@pytest.mark.parametrize(
    ("integrator", "minimum_order", "path_kind"),
    [
        ("rk4_fixed", 3.5, "rk4_dense"),
        ("exponential_midpoint", 1.8, "exponential_midpoint_reintegrated"),
    ],
)
def test_rarefied_lift_uniform_vorticity_shear_converges(
    tmp_path: Path,
    integrator: Literal["rk4_fixed", "exponential_midpoint"],
    minimum_order: float,
    path_kind: str,
) -> None:
    reference = _analytic_state(_END_TIME_S)
    errors: list[float] = []
    last_result = None

    for index, dt_s in enumerate((0.125, 0.0625, 0.03125)):
        case_path = _case(
            tmp_path / f"lift-{integrator}-{index}",
            integrator=integrator,
            dt_s=dt_s,
        )
        output = tmp_path / f"lift-result-{integrator}-{index}"
        simulate(load_case(case_path), output)
        last_result = open_result(output)
        final = last_result.read_final()
        np.testing.assert_array_equal(final.lifecycle, [1])
        state = np.concatenate((final.position_m[0], final.velocity_m_s[0]))
        errors.append(float(np.linalg.norm(state - reference)))

    assert last_result is not None
    assert errors[0] > errors[1] > errors[2] > 0.0
    observed_orders = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert float(np.min(observed_orders)) >= minimum_order
    assert last_result.manifest["resolved"]["physics_models"]["lift"] == {
        "model": "rarefied_vorticity_sensitivity",
        "revision": _REVISION,
        "lift_coefficient": _LIFT_COEFFICIENT,
    }
    assert last_result.manifest["resolved"]["path_kind"] == path_kind
    assert last_result.manifest["data_coordinate_system"] == "axisymmetric_rz"


def _case(
    directory: Path,
    *,
    integrator: Literal["rk4_fixed", "exponential_midpoint"],
    dt_s: float,
) -> Path:
    paths = materialize_microcase("C07", directory)
    case = load_case(paths.case_path)
    radial_shift = np.asarray([2.0, 0.0])
    source = replace(
        case.data.sources[0],
        position_m=_INITIAL_POSITION_M[None, :],
        velocity_m_s=_INITIAL_VELOCITY_M_S[None, :],
        charge_number=np.asarray([0.0]),
        mass_kg=np.asarray([_PARTICLE_MASS_KG]),
        drag_diameter_m=np.asarray([_PARTICLE_DIAMETER_M]),
        electrostatic_radius_m=np.asarray([0.0]),
        displaced_volume_m3=np.asarray([0.0]),
    )
    layout = RegularLayout(
        "gas",
        np.asarray([2.0, 3.0]),
        np.asarray([0.0, 1.0]),
        np.ones((1, 1), dtype=np.uint8),
    )
    scalar_components = ("value",)
    velocity_values = np.asarray(
        [
            [0.0, 0.0],
            [_SHEAR_RATE_S_INV, 0.0],
            [0.0, 0.0],
            [_SHEAR_RATE_S_INV, 0.0],
        ]
    )
    fields = (
        FieldData(
            "gas_velocity",
            "gas",
            "node",
            ("r", "z"),
            "axisymmetric_rz",
            velocity_values,
            "m/s",
        ),
        _constant_scalar_field("gas_density", _GAS_DENSITY_KG_M3, "kg/m^3"),
        _constant_scalar_field("gas_mean_free_path", _GAS_MEAN_FREE_PATH_M, "m"),
        FieldData(
            "azimuthal_vorticity",
            "gas",
            "node",
            scalar_components,
            "scalar",
            np.full((4, 1), _SHEAR_RATE_S_INV),
            "1/s",
        ),
    )
    data_path = paths.case_path.with_name("rarefied-vorticity-lift.h5")
    info = write(
        data_path,
        replace(
            case.data,
            coordinate_system="axisymmetric_rz",
            geometry=replace(
                case.data.geometry,
                nodes_m=case.data.geometry.nodes_m + radial_shift,
            ),
            layouts=(layout,),
            fields=fields,
            sources=(source,),
        ),
    )

    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["motion"]["mode"] = "axisymmetric_rz_meridional"
    document["time"] = {"start_s": 0.0, "end_s": _END_TIME_S, "dt_s": dt_s}
    document["solver"]["integrator"] = integrator
    document["physics"] = {
        "charge": {"model": "fixed"},
        "lift": {
            "model": "rarefied_vorticity_sensitivity",
            "revision": _REVISION,
            "gas_velocity_field": "gas_velocity",
            "gas_density_field": "gas_density",
            "gas_mean_free_path_field": "gas_mean_free_path",
            "azimuthal_gas_vorticity_field": "azimuthal_vorticity",
            "lift_coefficient": _LIFT_COEFFICIENT,
            "applicability": "error",
        },
    }
    document["output"] = {"trajectories": None}
    case_path = paths.case_path.with_name("rarefied-vorticity-lift.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _constant_scalar_field(name: str, value: float, unit: str) -> FieldData:
    return FieldData(
        name,
        "gas",
        "node",
        ("value",),
        "scalar",
        np.full((4, 1), value),
        unit,
    )


def _analytic_state(time_s: float) -> np.ndarray:
    coupling = (
        _LIFT_COEFFICIENT
        * math.pi
        * _GAS_DENSITY_KG_M3
        * _GAS_MEAN_FREE_PATH_M
        * _PARTICLE_RADIUS_M**2
        / _PARTICLE_MASS_KG
    )
    kappa_s_inv = coupling * _SHEAR_RATE_S_INV
    frequency_s_inv = math.sqrt(kappa_s_inv * (kappa_s_inv + _SHEAR_RATE_S_INV))
    conserved_velocity_m_s = _INITIAL_VELOCITY_M_S[0] + kappa_s_inv * _INITIAL_POSITION_M[1]
    equilibrium_z_m = conserved_velocity_m_s / (kappa_s_inv + _SHEAR_RATE_S_INV)
    phase = frequency_s_inv * time_s
    sin_phase = math.sin(phase)
    cos_phase = math.cos(phase)
    z_offset_m = _INITIAL_POSITION_M[1] - equilibrium_z_m
    z_m = (
        equilibrium_z_m
        + z_offset_m * cos_phase
        + _INITIAL_VELOCITY_M_S[1] * sin_phase / frequency_s_inv
    )
    velocity_z_m_s = (
        -z_offset_m * frequency_s_inv * sin_phase + _INITIAL_VELOCITY_M_S[1] * cos_phase
    )
    velocity_r_m_s = conserved_velocity_m_s - kappa_s_inv * z_m
    r_m = (
        _INITIAL_POSITION_M[0]
        + (conserved_velocity_m_s - kappa_s_inv * equilibrium_z_m) * time_s
        - kappa_s_inv * z_offset_m * sin_phase / frequency_s_inv
        - kappa_s_inv * _INITIAL_VELOCITY_M_S[1] * (1.0 - cos_phase) / frequency_s_inv**2
    )
    return np.asarray([r_m, z_m, velocity_r_m_s, velocity_z_m_s])
