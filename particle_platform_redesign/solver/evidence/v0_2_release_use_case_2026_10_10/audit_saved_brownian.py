"""Post-observation audit of saved release OU results; does not simulate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from tests.verification.ou_reference import (
    covariance_matrix,
    quadrature_covariance,
    spatial_affine_ou_moments,
    stationary_oml_forced_ou_mean,
    time_linear_ou_moments,
)

from chamber_particles import load_case, open_result

ROOT = Path(__file__).resolve().parent
SOLVER = ROOT.parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(name: str, output: str, case_name: str) -> dict:
    parent = next((ROOT / "pytest_b").glob(name))
    result_path = parent / output
    case_path = parent / case_name / "case.yaml"
    case = load_case(case_path)
    final = open_result(result_path).read_final()
    manifest = open_result(result_path).manifest
    fields = {field.name: field.values for field in case.data.fields}
    mass = float(final.mass_kg[0])
    diameter = float(final.drag_diameter_m[0])
    temperature = float(fields["gas_temperature"].reshape(-1)[0])
    rho = float(fields["gas_density"].reshape(-1)[0])
    thermal = 1.380649e-23 * temperature / mass
    rate = (
        math.pi
        * diameter**2
        * rho
        * math.sqrt(8 * 1.380649e-23 * temperature / (math.pi * 4.65e-26))
        / (3 * mass)
    )
    initial_velocity = np.asarray([0.2, -0.1])
    flow = np.asarray([0.1, -0.05])
    count = final.particle_id.size
    duration = float(final.time_s[0])
    refinement = []
    charge = None
    if case_name == "time-linear":
        arguments = (
            rate,
            3.0,
            thermal,
            duration,
            np.zeros(2),
            initial_velocity,
            flow,
            np.asarray([0.2, -0.1]),
        )
        fine = time_linear_ou_moments(*arguments, quadrature_order=64)
        coarse = time_linear_ou_moments(*arguments, quadrature_order=32)
        refinement = [float(np.max(np.abs(a - b))) for a, b in zip(coarse, fine, strict=True)]
        for a, b in zip(coarse, fine, strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-13, atol=1e-17)
        expected = [(np.asarray([fine[0][i], fine[1][i]]), fine[2]) for i in range(2)]
        mean_budget = np.asarray([1e-6, 1e-6])
    elif case_name == "spatial-affine":
        expected = []
        for i, gradient in enumerate((-3.0, 0.0)):
            arguments = (
                rate,
                gradient,
                thermal,
                duration,
                np.asarray([0.0, initial_velocity[i]]),
                0.0,
            )
            fine = spatial_affine_ou_moments(*arguments, quadrature_order=64)
            coarse = spatial_affine_ou_moments(*arguments, quadrature_order=32)
            refinement.append(
                [float(np.max(np.abs(a - b))) for a, b in zip(coarse, fine, strict=True)]
            )
            for a, b in zip(coarse, fine, strict=True):
                np.testing.assert_allclose(a, b, rtol=2e-13, atol=1e-17)
            expected.append(fine)
        mean_budget = np.asarray([1e-5, 3e-5])
    else:
        arguments = {
            "duration": duration,
            "rate": rate,
            "initial_charge": -17.25,
            "radius": float(final.electrostatic_radius_m[0]),
            "density": 1e8,
            "electron_temperature": 20000.0,
            "ion_temperature": 300.0,
            "ion_mass": 6.6335209e-26,
            "particle_mass": mass,
            "electric_field": np.asarray([10.0, 0.0]),
            "initial_velocity": initial_velocity,
            "flow": flow,
        }
        fine = stationary_oml_forced_ou_mean(**arguments, intervals=4096)
        coarse = stationary_oml_forced_ou_mean(**arguments, intervals=2048)
        refinement = [
            float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
            for a, b in zip(coarse, fine, strict=True)
        ]
        for a, b in zip(coarse, fine, strict=True):
            np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)
        covariance = covariance_matrix(quadrature_covariance(rate * duration, thermal, rate))
        expected = [(np.asarray([fine[1][i], fine[2][i]]), covariance) for i in range(2)]
        charge_error = float(np.max(np.abs(final.charge_number - fine[0])))
        charge = {
            "reference_charge_number": fine[0],
            "maximum_absolute_error": charge_error,
            "budget": 2e-4,
            "negative_branch": bool((final.charge_number < 0).all()),
            "pass": charge_error <= 2e-4,
        }
        assert charge["pass"] and charge["negative_branch"]
        mean_budget = np.asarray([2e-8, 5e-7])
    components = []
    for i, (mean, covariance) in enumerate(expected):
        sample = np.column_stack((final.position_m[:, i], final.velocity_m_s[:, i]))
        observed_mean = sample.mean(axis=0)
        mean_error = np.abs(observed_mean - mean)
        mean_mc_allowance = 7 * np.sqrt(np.diag(covariance) / count)
        observed_covariance = np.cov(sample, rowvar=False, bias=case_name == "time-linear")
        covariance_error = np.abs(observed_covariance - covariance)
        if case_name == "time-linear":
            covariance_mc_allowance = np.zeros((2, 2))
            covariance_budget = 0.05 * np.abs(covariance) + 2e-9
        else:
            covariance_mc_allowance = 7 * np.sqrt(
                (covariance**2 + np.outer(np.diag(covariance), np.diag(covariance))) / (count - 1)
            )
            covariance_budget = (0.003 if case_name == "spatial-affine" else 0) * np.sqrt(
                np.outer(np.diag(covariance), np.diag(covariance))
            )
        gate = mean_mc_allowance + mean_budget
        cov_gate = covariance_mc_allowance + covariance_budget
        passed = bool(np.all(mean_error <= gate) and np.all(covariance_error <= cov_gate))
        assert passed
        components.append(
            {
                "component": i,
                "reference_mean": mean.tolist(),
                "observed_mean": observed_mean.tolist(),
                "mean_absolute_error": mean_error.tolist(),
                "seven_standard_error_allowance": mean_mc_allowance.tolist(),
                "fixed_mean_budget": mean_budget.tolist(),
                "maximum_mean_error_to_gate": float(np.max(mean_error / gate)),
                "reference_covariance": covariance.tolist(),
                "observed_covariance": observed_covariance.tolist(),
                "covariance_absolute_error": covariance_error.tolist(),
                "covariance_mc_allowance": covariance_mc_allowance.tolist(),
                "fixed_covariance_allowance": covariance_budget.tolist(),
                "maximum_covariance_error_to_gate": float(np.max(covariance_error / cov_gate)),
                "pass": passed,
            }
        )
    assert manifest["counts"]["failure_events"] == manifest["counts"]["boundary_events"] == 0
    assert bool(final.kinematics_valid.all()) and bool((final.lifecycle == 1).all())
    return {
        "count": count,
        "duration_s": duration,
        "derived_rate0_s_inv": rate,
        "thermal_velocity_variance": thermal,
        "counts": manifest["counts"],
        "engine_revision": manifest["engine_algorithm_revision"],
        "event_revision": manifest["event_algorithm_revision"],
        "data_content_hash": manifest["data_content_hash"],
        "reference_refinement_absolute_difference": refinement,
        "components": components,
        "charge": charge,
        "confidence_scope": "Inherited seven-SE or fixed-relative covariance regression criteria; no newly registered formal joint confidence family.",
        "raw_artifacts_sha256": {
            str(p.relative_to(SOLVER)).replace("\\", "/"): digest(p)
            for p in parent.rglob("*")
            if p.is_file()
        },
        "pass": True,
    }


if __name__ == "__main__":
    report = {
        "artifact": "saved_release_brownian_independent_reaggregation",
        "scientific_simulations_performed": 0,
        "B01_TIME_LINEAR_OU": audit(
            "test_brownian_time_linear_drag0", "time-linear-result", "time-linear"
        ),
        "B02_SPATIAL_AFFINE_OU": audit(
            "test_brownian_spatial_affine_f0", "spatial-affine-result", "spatial-affine"
        ),
        "B03_STATIONARY_OML_COULOMB_OU": audit(
            "test_brownian_stationary_oml_a0", "oml-coulomb-result", "oml-coulomb"
        ),
    }
    (ROOT / "b_independent_observations.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: {
                    "mean_ratio": max(c["maximum_mean_error_to_gate"] for c in v["components"]),
                    "covariance_ratio": max(
                        c["maximum_covariance_error_to_gate"] for c in v["components"]
                    ),
                    "charge": v["charge"],
                }
                for k, v in report.items()
                if k.startswith("B")
            },
            ensure_ascii=True,
        )
    )
