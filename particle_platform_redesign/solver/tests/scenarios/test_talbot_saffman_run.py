from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import BoundaryData, FieldData, RegularLayout, write
from tests.verification.microcases import materialize_microcase


def test_talbot_and_saffman_compose_in_one_public_run(tmp_path: Path) -> None:
    case_path = _case(tmp_path / "talbot-saffman")
    output = tmp_path / "result"

    simulate(load_case(case_path), output)
    result = open_result(output)
    final = result.read_final()

    assert bool(np.isfinite(final.position_m).all())
    assert bool(np.isfinite(final.velocity_m_s).all())
    assert result.manifest["resolved"]["physics_models"]["thermophoresis"] == {
        "model": "talbot",
        "revision": "talbot_cross_regime_radius_knudsen_v1",
        "particle_thermal_conductivity_W_m_K": 0.2,
        "thermal_slip_coefficient": 1.17,
        "momentum_exchange_coefficient": 1.14,
        "thermal_exchange_coefficient": 2.18,
    }
    assert result.manifest["resolved"]["physics_models"]["lift"] == {
        "model": "saffman",
        "revision": "saffman_unbounded_creeping_shear_v1",
    }


@pytest.mark.parametrize(
    ("integrator", "dt_s", "absolute_tolerance"),
    [("rk4_fixed", 0.05, 2.0e-12), ("exponential_midpoint", 0.01, 2.0e-9)],
)
def test_talbot_in_affine_temperature_matches_independent_exact_motion(
    tmp_path: Path,
    integrator: str,
    dt_s: float,
    absolute_tolerance: float,
) -> None:
    case_path = _case(tmp_path / integrator)
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["time"] = {"start_s": 0.0, "end_s": 1.0, "dt_s": dt_s}
    document["solver"]["integrator"] = integrator
    document["physics"] = {
        "charge": {"model": "fixed"},
        "thermophoresis": _physics()["thermophoresis"],
    }
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    # T(x)=300+10*x, v(0)=0; the original radius formula gives x''=-B/T(x).
    radius, mass, viscosity, density, gradient = 1.0e-4, 1.0e-8, 1.0e-3, 1.0, 10.0
    knudsen, ratio = 5.0e-6 / radius, 0.03 / 0.2
    correction = 1.17 * (ratio + 2.18 * knudsen)
    correction /= (1.0 + 3.0 * 1.14 * knudsen) * (1.0 + 2.0 * ratio + 2.0 * 2.18 * knudsen)
    b = 12.0 * math.pi * radius * viscosity**2 * correction * gradient / (mass * density)
    initial_temperature = 300.0 + gradient * 0.25
    # Energy integration: v=-sqrt(2*B/G)*w, T/T_initial=exp(-w*w),
    # erf(w)=t*sqrt(2*B*G/pi)/T_initial. Invert only this scalar analytic relation.
    target = math.sqrt(2.0 * b * gradient / math.pi) / initial_temperature
    lower, upper = 0.0, 1.0
    for _ in range(64):
        middle = (lower + upper) / 2.0
        if math.erf(middle) < target:
            lower = middle
        else:
            upper = middle
    w = (lower + upper) / 2.0
    exact_position = np.asarray([0.25 + initial_temperature * math.expm1(-w * w) / gradient, 0.25])
    exact_velocity = np.asarray([-math.sqrt(2.0 * b / gradient) * w, 0.0])

    output = tmp_path / f"{integrator}-result"
    simulate(load_case(case_path), output)
    final = open_result(output).read_final()
    np.testing.assert_allclose(
        final.position_m[0], exact_position, rtol=0.0, atol=absolute_tolerance
    )
    np.testing.assert_allclose(
        final.velocity_m_s[0], exact_velocity, rtol=0.0, atol=absolute_tolerance
    )
    np.testing.assert_array_equal(final.lifecycle, [1])


def test_saffman_rotation_converges_to_linear_exact_solution(tmp_path: Path) -> None:
    errors: list[float] = []
    end_s = 0.1
    speed_m_s = 1.0e-3
    coupling_rate_s_inv = 6.46 * (1.0e-4) ** 2 * np.sqrt(1.0e-3 * 1.0 * 100.0) / 1.0e-8
    angle = coupling_rate_s_inv * end_s
    exact_velocity = np.asarray([speed_m_s * np.cos(angle), speed_m_s * np.sin(angle)])
    exact_position = np.asarray(
        [
            0.25 + speed_m_s * np.sin(angle) / coupling_rate_s_inv,
            0.25 + speed_m_s * (1.0 - np.cos(angle)) / coupling_rate_s_inv,
        ]
    )

    for label, dt_s in (("coarse", 0.02), ("fine", 0.01)):
        case_path = _saffman_rotation_case(
            tmp_path / label,
            dt_s=dt_s,
            speed_m_s=speed_m_s,
        )
        output = tmp_path / f"{label}-result"
        simulate(load_case(case_path), output)
        final = open_result(output).read_final()
        error = np.linalg.norm(final.velocity_m_s[0] - exact_velocity)
        error += np.linalg.norm(final.position_m[0] - exact_position)
        errors.append(float(error))

    assert errors[1] < errors[0] / 10.0


def test_saffman_hierarchy_violation_fails_closed(tmp_path: Path) -> None:
    case_path = _saffman_rotation_case(
        tmp_path / "inapplicable",
        dt_s=0.01,
        speed_m_s=0.1,
    )
    output = tmp_path / "inapplicable-result"

    simulate(load_case(case_path), output)
    result = open_result(output)
    final = result.read_final()
    failures = result.read_failure_events()
    reason = result.manifest["failure_reason_codes"]["model_applicability"]
    np.testing.assert_array_equal(final.lifecycle, [4])
    np.testing.assert_array_equal(final.failure_reason_code, [reason])
    np.testing.assert_array_equal(failures.reason_code, [reason])


def _case(directory: Path) -> Path:
    paths = materialize_microcase("C07", directory)
    base = load_case(paths.case_path)
    source = replace(
        base.data.sources[0],
        position_m=np.asarray([[0.25, 0.25]]),
        velocity_m_s=np.asarray([[0.0, 0.0]]),
        charge_number=np.zeros(1),
        mass_kg=np.asarray([1.0e-8]),
        drag_diameter_m=np.asarray([2.0e-4]),
        electrostatic_radius_m=np.zeros(1),
        displaced_volume_m3=np.zeros(1),
    )
    layout = RegularLayout(
        "gas",
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, 1.0]),
        np.ones((1, 1), dtype=np.uint8),
    )
    fields = (
        _vector_field("gas_velocity", ((0.0, 0.0),) * 4, "m/s"),
        _scalar_field("gas_density", (1.0, 1.0, 1.0, 1.0), "kg/m^3"),
        _scalar_field("gas_viscosity", (1.0e-3,) * 4, "Pa*s"),
        _scalar_field("mean_free_path", (5.0e-6,) * 4, "m"),
        _scalar_field("temperature", (300.0, 300.0, 310.0, 310.0), "K"),
        _vector_field("temperature_gradient", ((10.0, 0.0),) * 4, "K/m"),
        _scalar_field("gas_conductivity", (0.03,) * 4, "W/(m*K)"),
        _scalar_field("vorticity", (100.0,) * 4, "1/s"),
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
    data_path = paths.case_path.with_name("talbot-saffman.h5")
    info = write(
        data_path,
        replace(
            base.data,
            geometry=replace(base.data.geometry, boundary=boundaryless, group_names=()),
            layouts=(layout,),
            fields=fields,
            sources=(source,),
        ),
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"] = {"start_s": 0.0, "end_s": 0.005, "dt_s": 0.0005}
    document["solver"]["integrator"] = "exponential_midpoint"
    document["physics"] = _physics()
    document["boundaries"] = []
    document["output"] = {"trajectories": None}
    case_path = paths.case_path.with_name("talbot-saffman.yaml")
    case_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return case_path


def _saffman_rotation_case(directory: Path, *, dt_s: float, speed_m_s: float) -> Path:
    case_path = _case(directory)
    case = load_case(case_path)
    source = replace(
        case.data.sources[0],
        velocity_m_s=np.asarray([[speed_m_s, 0.0]]),
    )
    data_path = case_path.with_name("saffman-rotation.h5")
    info = write(data_path, replace(case.data, sources=(source,)))
    document = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    document["case"]["data_path"] = data_path.name
    document["case"]["expected_content_hash"] = info.content_hash
    document["time"] = {"start_s": 0.0, "end_s": 0.1, "dt_s": dt_s}
    document["solver"]["integrator"] = "rk4_fixed"
    document["physics"] = {
        "charge": {"model": "fixed"},
        "lift": _physics()["lift"],
    }
    rotation_path = case_path.with_name("saffman-rotation.yaml")
    rotation_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return rotation_path


def _physics() -> dict[str, object]:
    return {
        "charge": {"model": "fixed"},
        "drag": {
            "model": "stokes_cunningham",
            "revision": "stokes_cunningham_allen_raabe_air_v1",
            "gas_velocity_field": "gas_velocity",
            "gas_density_field": "gas_density",
            "gas_dynamic_viscosity_field": "gas_viscosity",
            "gas_mean_free_path_field": "mean_free_path",
            "applicability": "error",
        },
        "thermophoresis": {
            "model": "talbot",
            "revision": "talbot_cross_regime_radius_knudsen_v1",
            "gas_temperature_field": "temperature",
            "gas_temperature_gradient_field": "temperature_gradient",
            "gas_density_field": "gas_density",
            "gas_dynamic_viscosity_field": "gas_viscosity",
            "gas_thermal_conductivity_field": "gas_conductivity",
            "gas_mean_free_path_field": "mean_free_path",
            "particle_thermal_conductivity_W_m_K": 0.2,
            "thermal_slip_coefficient": 1.17,
            "momentum_exchange_coefficient": 1.14,
            "thermal_exchange_coefficient": 2.18,
            "applicability": "error",
        },
        "lift": {
            "model": "saffman",
            "revision": "saffman_unbounded_creeping_shear_v1",
            "gas_velocity_field": "gas_velocity",
            "gas_density_field": "gas_density",
            "gas_dynamic_viscosity_field": "gas_viscosity",
            "gas_mean_free_path_field": "mean_free_path",
            "out_of_plane_gas_vorticity_field": "vorticity",
            "applicability": "error",
        },
    }


def _scalar_field(name: str, values: tuple[float, ...], unit: str) -> FieldData:
    return FieldData(
        name,
        "gas",
        "node",
        ("value",),
        "scalar",
        np.asarray(values, dtype=np.float64)[:, None],
        unit,
    )


def _vector_field(
    name: str,
    values: tuple[tuple[float, float], ...],
    unit: str,
) -> FieldData:
    return FieldData(
        name,
        "gas",
        "node",
        ("x", "y"),
        "cartesian_xy",
        np.asarray(values, dtype=np.float64),
        unit,
    )
