from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from chamber_particles import load_case, open_result, simulate
from chamber_particles.case_format import RealizedTableSource, write
from tests.verification.microcases import materialize_microcase
from tests.verification.ou_reference import quadrature_covariance


def _prescribed_case(directory: Path, count: int) -> Path:
    paths = materialize_microcase("C02", directory)
    case = load_case(paths.case_path)
    source = case.data.sources[0]
    assert isinstance(source, RealizedTableSource)
    mass = float(source.mass_kg[0])
    properties = {
        "mass_kg": mass,
        "drag_diameter_m": float(source.drag_diameter_m[0]),
        "electrostatic_radius_m": float(source.electrostatic_radius_m[0]),
        "displaced_volume_m3": float(source.displaced_volume_m3[0]),
    }
    fields = {field.name: field for field in case.data.fields}
    scalar = fields["gas_temperature"]
    vector = fields["gas_velocity"]
    coefficients = {
        "friction": (2.0 * mass, "kg/s"),
        "D": (4.0 * mass * mass * 1.0e-4, "N^2*s"),
        "offset": (0.0, "1"),
        "alpha": (0.5, "1"),
        "Ai": (0.4, "1/s"),
        "betaZ": (2.0, "1"),
    }
    added = [
        replace(scalar, name=name, unit=unit, values=np.full_like(scalar.values, value))
        for name, (value, unit) in coefficients.items()
    ]
    electric = mass / 1.602176634e-19 * np.asarray([0.1, -0.05])
    added.append(
        replace(
            vector,
            name="E",
            unit="V/m",
            values=np.broadcast_to(electric, vector.values.shape).copy(),
        )
    )
    source = replace(
        source,
        particle_id=np.arange(1, count + 1, dtype="<i8"),
        release_time_s=np.zeros(count),
        position_m=np.zeros((count, 2)),
        velocity_m_s=np.tile([0.2, -0.1], (count, 1)),
        charge_number=np.full(count, 3.0),
        mass_kg=np.full(count, mass),
        drag_diameter_m=np.full(count, properties["drag_diameter_m"]),
        electrostatic_radius_m=np.full(count, properties["electrostatic_radius_m"]),
        displaced_volume_m3=np.full(count, properties["displaced_volume_m3"]),
        contact_radius_m=np.zeros(count),
        model_weight=np.ones(count),
        material_id=np.zeros(count, dtype="<i4"),
    )
    data_path = directory / "prescribed.h5"
    info = write(
        data_path, replace(case.data, fields=(*case.data.fields, *added), sources=(source,))
    )
    document = yaml.safe_load(paths.case_path.read_text(encoding="utf-8"))
    document["case"].update(data_path=data_path.name, expected_content_hash=info.content_hash)
    document["time"] = {"start_s": 0.0, "end_s": 0.2, "dt_s": 0.01}
    document["solver"].update(integrator="ou_langevin", seed=1741)
    document["physics"] = {
        "charge": {
            "model": "prescribed_exponential_linear",
            "revision": "prescribed_exponential_linear_collection_v1",
            "electron_rate_reference_number_s": 0.0,
            "electron_exponent_offset_field": "offset",
            "electron_charge_slope_field": "alpha",
            "ion_rate_field": "Ai",
            "ion_charge_slope_field": "betaZ",
            "particle_properties": properties,
        },
        "drag": {
            "model": "prescribed_linear_friction",
            "revision": "prescribed_linear_friction_v1",
            "gas_velocity_field": "gas_velocity",
            "friction_coefficient_field": "friction",
            "particle_properties": properties,
        },
        "noise": {
            "model": "prescribed_force_diffusion",
            "revision": "prescribed_force_diffusion_midpoint_2d_v1",
            "force_spectral_density_field": "D",
            "interval_tree_depth": 1,
            "adaptive_max_depth": 3,
            "particle_properties": properties,
        },
        "electric": {"model": "coulomb", "revision": "electric_coulomb_v1", "electric_field": "E"},
    }
    document["output"]["trajectories"] = None
    path = directory / "prescribed.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def test_prescribed_charge_electric_diffusion_matches_independent_ou_and_scalar_limits(
    tmp_path: Path,
) -> None:
    count, time, gamma, kappa, theta = 12_000, 0.2, 2.0, 0.8, 1.0e-4
    path = _prescribed_case(tmp_path / "ensemble", count)
    output = tmp_path / "result"
    simulate(load_case(path), output)
    result = open_result(output)
    final = result.read_final()
    b, c = math.exp(-gamma * time), math.exp(-kappa * time)
    memory = (1.0 - b) / gamma
    u, v, force = np.asarray([0.1, -0.05]), np.asarray([0.2, -0.1]), np.asarray([0.1, -0.05])
    mean_v = u + (v - u) * b + force * (0.5 * memory + 2.5 * (c - b) / (gamma - kappa))
    mean_x = (
        u * time
        + (v - u) * memory
        + force
        * (0.5 * (time - memory) / gamma + 2.5 * ((1.0 - c) / kappa - memory) / (gamma - kappa))
    )
    qxx, qxv, qvv = quadrature_covariance(gamma * time, theta, gamma)
    covariance = np.asarray([[qxx, qxv], [qxv, qvv]])
    for component in range(2):
        sample = np.column_stack((final.position_m[:, component], final.velocity_m_s[:, component]))
        expected = np.asarray([mean_x[component], mean_v[component]])
        allowance = 7.0 * np.sqrt(np.diag(covariance) / count)
        assert np.all(np.abs(sample.mean(axis=0) - expected) < allowance)
        np.testing.assert_allclose(
            np.cov(sample, rowvar=False, bias=True), covariance, rtol=0.06, atol=0.0
        )
    np.testing.assert_allclose(final.charge_number, 0.5 + 2.5 * c, rtol=1.0e-14)
    counts = result.manifest["failure_reason_counts"]
    assert not any(counts.values())
    resolved = result.manifest["resolved"]
    assert (
        resolved["physics_models"]["noise"]["revision"]
        == "prescribed_force_diffusion_midpoint_2d_v1"
    )


def test_prescribed_noise_observations_and_workers_preserve_accepted_state(tmp_path: Path) -> None:
    path = _prescribed_case(tmp_path / "identity", 64)
    plain = tmp_path / "plain"
    simulate(load_case(path), plain)
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["output"]["trajectories"] = {
        "selection": "all",
        "schedule": {"explicit_times_s": [0.0, 0.037, 0.199, 0.2]},
    }
    framed_path = path.with_name("framed.yaml")
    framed_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    framed = tmp_path / "framed"
    simulate(load_case(framed_path), framed, workers=2)
    left, right = open_result(plain).read_final(), open_result(framed).read_final()
    for name in ("position_m", "velocity_m_s", "charge_number", "lifecycle"):
        np.testing.assert_array_equal(getattr(left, name), getattr(right, name))


@pytest.mark.parametrize("integrator,order", [("rk4_fixed", 3.5), ("exponential_midpoint", 1.8)])
def test_prescribed_coupled_deterministic_models_converge_to_matrix_solution(
    tmp_path: Path, integrator: str, order: float
) -> None:
    path = _prescribed_case(tmp_path / integrator, 1)
    case = load_case(path)
    source = case.data.sources[0]
    mass = float(source.mass_kg[0])
    scalar = next(field for field in case.data.fields if field.name == "gas_temperature")
    vector = next(field for field in case.data.fields if field.name == "gas_velocity")
    added = [
        replace(
            vector,
            name=name,
            unit="N",
            values=np.broadcast_to(mass * np.asarray(value), vector.values.shape).copy(),
        )
        for name, value in (("F0", [0.03, 0.04]), ("F1", [-0.02, 0.01]), ("F2", [0.0, 0.0]))
    ]
    added.append(
        replace(
            scalar, name="rotation", unit="kg/s", values=np.full_like(scalar.values, 0.5 * mass)
        )
    )
    data_path = path.with_name("forced.h5")
    info = write(data_path, replace(case.data, fields=(*case.data.fields, *added)))
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["case"].update(data_path=data_path.name, expected_content_hash=info.content_hash)
    document["solver"]["integrator"] = integrator
    del document["physics"]["noise"]
    properties = document["physics"]["drag"]["particle_properties"]
    document["physics"]["ion_drag"] = {
        "model": "prescribed_charge_polynomial",
        "revision": "prescribed_charge_polynomial_v1",
        "constant_force_field": "F0",
        "linear_charge_force_field": "F1",
        "quadratic_charge_force_field": "F2",
        "particle_properties": properties,
    }
    document["physics"]["lift"] = {
        "model": "prescribed_rotational_lift",
        "revision": "prescribed_rotational_lift_v1",
        "gas_velocity_field": "gas_velocity",
        "rotation_coefficient_field": "rotation",
        "particle_properties": properties,
    }
    expected_position, expected_velocity = _matrix_reference_solution(0.2)
    errors = []
    for index, dt in enumerate((0.02, 0.01, 0.005)):
        document["time"]["dt_s"] = dt
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        output = tmp_path / f"result-{index}"
        simulate(load_case(path), output)
        final = open_result(output).read_final()
        errors.append(
            float(
                np.linalg.norm(
                    np.concatenate(
                        (
                            final.position_m[0] - expected_position,
                            final.velocity_m_s[0] - expected_velocity,
                        )
                    )
                )
            )
        )
        np.testing.assert_allclose(
            final.charge_number, 0.5 + 2.5 * math.exp(-0.8 * 0.2), atol=1.0e-9
        )
    assert math.log2(errors[0] / errors[1]) >= order
    assert math.log2(errors[1] / errors[2]) >= order


def _matrix_reference_solution(time: float) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the independent constant 2x2 rotation/relaxation system."""

    matrix = np.asarray([[-2.0, -0.5], [0.5, -2.0]])
    identity = np.eye(2)
    exponential = math.exp(-2.0 * time) * np.asarray(
        [
            [math.cos(0.5 * time), -math.sin(0.5 * time)],
            [math.sin(0.5 * time), math.cos(0.5 * time)],
        ]
    )
    memory = np.linalg.solve(matrix, exponential - identity)
    target = np.asarray([0.1, -0.05])
    initial_relative = np.asarray([0.2, -0.1]) - target
    forcing = np.asarray([0.08, -0.04])
    steady = np.asarray([0.03, 0.04]) + 0.5 * forcing
    decay = math.exp(-0.8 * time)
    velocity = target + exponential @ initial_relative + memory @ steady
    velocity += np.linalg.solve(matrix + 0.8 * identity, exponential - decay * identity) @ (
        2.5 * forcing
    )
    position = (
        target * time
        + memory @ initial_relative
        + np.linalg.solve(matrix, memory - time * identity) @ steady
    )
    position += np.linalg.solve(
        matrix + 0.8 * identity, memory - (1.0 - decay) / 0.8 * identity
    ) @ (2.5 * forcing)
    return position, velocity


def test_coulomb_cap_crossing_preserves_dynamic_charge_and_quadratic_ion_feedback(
    tmp_path: Path,
) -> None:
    path = _prescribed_case(tmp_path / "cap-crossing", 1)
    case = load_case(path)
    source = case.data.sources[0]
    mass = float(source.mass_kg[0])
    source = replace(source, charge_number=np.asarray([-3.0]), velocity_m_s=np.zeros((1, 2)))
    fields = []
    for field in case.data.fields:
        if field.name == "Ai":
            field = replace(field, values=np.full_like(field.values, 4.0))
        elif field.name == "E":
            field = replace(
                field, values=np.tile([mass / 1.602176634e-19, 0.0], (field.values.shape[0], 1))
            )
        fields.append(field)
    vector = next(field for field in fields if field.name == "gas_velocity")
    for name, value in (("F0", 0.0), ("F1", mass), ("F2", mass)):
        fields.append(
            replace(
                vector,
                name=name,
                unit="N",
                values=np.tile([value, 0.0], (vector.values.shape[0], 1)),
            )
        )
    info = write(
        path.with_name("cap.h5"), replace(case.data, sources=(source,), fields=tuple(fields))
    )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    document["case"].update(data_path="cap.h5", expected_content_hash=info.content_hash)
    document["time"].update(end_s=0.25, dt_s=0.00025)
    document["solver"]["integrator"] = "exponential_midpoint"
    del document["physics"]["noise"]
    properties = document["physics"].pop("drag")["particle_properties"]
    document["physics"]["electric"].update(
        revision="electric_coulomb_charge_cap_v1", maximum_abs_charge_number=1.0
    )
    document["physics"]["ion_drag"] = {
        "model": "prescribed_charge_polynomial",
        "revision": "prescribed_charge_polynomial_v1",
        "constant_force_field": "F0",
        "linear_charge_force_field": "F1",
        "quadratic_charge_force_field": "F2",
        "particle_properties": properties,
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    output = tmp_path / "result"
    simulate(load_case(path), output)
    result = open_result(output)
    final = result.read_final()
    position, velocity, charge = _capped_quadratic_reference(0.25)
    # Midpoint quadrature: sup |a''|<=2688 before the cap crossing, plus
    # one slope jump <=12. At h=.00025 these bounds allow <4e-6 m/s and
    # <1e-6 m in this .25 s interval, independently of measured errors.
    np.testing.assert_allclose(final.charge_number, [charge], rtol=0, atol=1e-13)
    np.testing.assert_allclose(final.velocity_m_s[0], [velocity, 0], rtol=0, atol=4e-6)
    np.testing.assert_allclose(final.position_m[0], [position, 0], rtol=0, atol=1e-6)
    assert -1 < final.charge_number[0] < 1
    failures = result.manifest["failure_reason_counts"]
    assert not any(failures.values())


def _capped_quadratic_reference(time: float) -> tuple[float, float, float]:
    """Post-crossing integral of Z'=4(1-2Z), a=clip(Z,-1,1)+Z+Z^2, Z0=-3."""
    kappa = 8.0
    crossing = math.log(3.5 / 1.5) / kappa
    decay, twice_decay = math.exp(-kappa * time), math.exp(-2 * kappa * time)
    at_crossing = math.exp(-kappa * crossing)
    ion_velocity = 0.75 * time - 7 * (1 - decay) / kappa + 12.25 * (1 - twice_decay) / (2 * kappa)
    ion_position = 0.375 * time * time - 7 * (time / kappa - (1 - decay) / kappa**2)
    ion_position += 12.25 * (time / (2 * kappa) - (1 - twice_decay) / (4 * kappa**2))
    electric_velocity = 0.5 * time - 1.5 * crossing - 3.5 * (at_crossing - decay) / kappa
    electric_position = -time * crossing + 0.5 * crossing**2 + 0.25 * (time - crossing) ** 2
    electric_position -= 3.5 * (
        at_crossing * (time - crossing) / kappa - (at_crossing - decay) / kappa**2
    )
    return ion_position + electric_position, ion_velocity + electric_velocity, 0.5 - 3.5 * decay
