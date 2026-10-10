"""Reproduce the limited numerical review checks; this is a review artifact.

Run from particle_platform_redesign/solver:
    uv run --locked python ../reviews/theory_numerics_independent_checks_2026-10-09.py

Uses the current solver for candidate calculations, not a historical environment.
It does not execute the public simulation workflow or a test suite.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np

from chamber_particles.integrators import DynamicsEvaluation, rk4_step
from chamber_particles.rng import philox4x32_10
from chamber_particles.stochastic import joint_ou_covariance


def green_covariance(order: int) -> np.ndarray:
    """Integrate the time-dependent OU Green kernels without candidate formulas."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    gamma0, alpha, theta, horizon = 1.7, 0.8, 0.7, 1.0
    s = horizon * (nodes + 1.0) / 2.0
    outer_weights = horizon * weights / 2.0
    q = s[:, None] + (horizon - s[:, None]) * (nodes[None, :] + 1.0) / 2.0
    response_qs = np.exp(
        -gamma0 * (q - s[:, None])
        - 0.5 * gamma0 * alpha * (q * q - s[:, None] ** 2)
    )
    displacement_response = np.sum(
        response_qs * weights[None, :] * (horizon - s[:, None]) / 2.0,
        axis=1,
    )
    velocity_response = np.exp(
        -gamma0 * (horizon - s)
        - 0.5 * gamma0 * alpha * (horizon * horizon - s * s)
    )
    kernel = np.column_stack((displacement_response, velocity_response))
    noise_variance_rate = 2.0 * theta * gamma0 * (1.0 + alpha * s)
    return (kernel.T * (outer_weights * noise_variance_rate)) @ kernel


def midpoint_covariance_checks() -> dict[str, object]:
    reference = green_covariance(96)
    refined_reference = green_covariance(192)
    rows = []
    errors = []
    for count in (4, 8, 16, 32, 64):
        duration = 1.0 / count
        covariance = np.zeros((2, 2))
        for index in range(count):
            rate = 1.7 * (1.0 + 0.8 * (index + 0.5) * duration)
            displacement = -math.expm1(-rate * duration) / rate
            transition = np.array(
                [[1.0, displacement], [0.0, math.exp(-rate * duration)]]
            )
            qxx, qxv, qvv = joint_ou_covariance(
                np.array([rate]), np.array([0.7]), np.array([duration])
            )
            increment_covariance = np.array([[qxx[0], qxv[0]], [qxv[0], qvv[0]]])
            covariance = transition @ covariance @ transition.T + increment_covariance
        error = float(np.max(np.abs(covariance - reference)))
        errors.append(error)
        rows.append(
            {"interval_count": count, "covariance": covariance.tolist(), "max_abs_error": error}
        )
    return {
        "scope": (
            "Linear no-wall system dx=v dt, dv=-gamma(t)v dt+sqrt(2*Theta*gamma(t)) dW; "
            "zero deterministic initial state; gamma(t)=1.7*(1+0.8*t), Theta=0.7, t=1. "
            "Nonzero stochastic covariance is propagated analytically; no sampled ensemble, "
            "public workflow, spatially varying coefficients, charge or first passage is tested."
        ),
        "reference_method": "Nested Gauss-Legendre integration of independent Green kernels",
        "reference_quadrature_orders": [96, 192],
        "reference_covariance": reference.tolist(),
        "reference_max_abs_difference_96_192": float(
            np.max(np.abs(reference - refined_reference))
        ),
        "candidate": "Midpoint-frozen transition with current joint_ou_covariance",
        "rows": rows,
        "observed_orders": [math.log2(left / right) for left, right in zip(errors, errors[1:])],
        "current_code": "solver/src/chamber_particles/stochastic.py:34",
        "primary_source": "https://doi.org/10.1103/PhysRevE.54.2084",
    }


def rng_checks() -> dict[str, object]:
    vectors = [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        (
            (0xFFFFFFFF,) * 4,
            (0xFFFFFFFF,) * 2,
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        ),
        (
            (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344),
            (0xA4093822, 0x299F31D0),
            (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1),
        ),
    ]
    rows = []
    for counter, key, expected in vectors:
        actual = philox4x32_10(counter, key)
        rows.append(
            {
                "counter": [hex(word) for word in counter],
                "key": [hex(word) for word in key],
                "expected": [hex(word) for word in expected],
                "actual": [hex(word) for word in actual],
                "exact_match": actual == expected,
            }
        )
    return {
        "scope": "Three official Philox4x32-10 known-answer vectors; no statistical battery",
        "current_code": "solver/src/chamber_particles/rng.py:233",
        "primary_source": "https://raw.githubusercontent.com/DEShawResearch/random123/main/tests/kat_vectors",
        "rows": rows,
        "all_exact_match": all(row["exact_match"] for row in rows),
    }


def harmonic_evaluator(ids, time, position, velocity, charge):
    del time, velocity
    return DynamicsEvaluation(
        -position,
        np.zeros_like(charge),
        np.ones(ids.size, dtype=bool),
        np.ones(ids.size, dtype=bool),
        np.zeros(ids.size, dtype=np.uint8),
    )


def dense_checks() -> dict[str, object]:
    duration, parameter = 0.2, 0.37
    start_position = np.array([[1.0, 0.0]])
    start_velocity = np.array([[0.0, 0.0]])
    proposal = rk4_step(
        np.array([0], dtype=np.int64),
        np.array([0.0]),
        np.array([duration]),
        start_position,
        start_velocity,
        np.array([0.0]),
        requires_stage_evaluation=True,
        evaluator=harmonic_evaluator,
    )
    sample = proposal.rk4_dense_state_at_rows(
        np.array([0], dtype=np.int64), np.array([parameter * duration])
    )
    end_position = proposal.end_position()
    end_velocity = proposal.end_velocity()
    derivative = (
        6.0 * parameter * (parameter - 1.0) * start_position / duration
        + 6.0 * parameter * (1.0 - parameter) * end_position / duration
        + (3.0 * parameter**2 - 4.0 * parameter + 1.0) * start_velocity
        + (3.0 * parameter**2 - 2.0 * parameter) * end_velocity
    )
    dense_velocity = float(sample.velocity_m_s[0, 0])
    position_derivative = float(derivative[0, 0])
    return {
        "scope": "One harmonic-oscillator RK4 proposal x''=-x; x(0)=1,v(0)=0; h=0.2,theta=0.37",
        "current_code": "solver/src/chamber_particles/integrators.py:3545",
        "primary_source": "https://www.dgp.toronto.edu/~karan/courses/csc418/fall_2002/notes/curves.html",
        "dense_position": float(sample.position_m[0, 0]),
        "dense_velocity": dense_velocity,
        "analytic_derivative_of_endpoint_Hermite_position": position_derivative,
        "kinematic_interior_difference": dense_velocity - position_derivative,
        "true_harmonic_velocity": -math.sin(parameter * duration),
        "interpretation": (
            "Position uses endpoint position/velocity Hermite cubic; velocity uses a separate "
            "first/fourth-stage-rate cubic. Interior dx/dt need not equal dense velocity exactly. "
            "This confirms interpolation semantics, not a new production bug or general error order."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    payload = {
        "review_date": "2026-10-09",
        "artifact_kind": "Reproducible limited numerical review calculations",
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "execution": "uv run --locked python ../reviews/theory_numerics_independent_checks_2026-10-09.py",
        "ou_covariance": midpoint_covariance_checks(),
        "philox_known_answers": rng_checks(),
        "rk4_dense_semantics": dense_checks(),
        "existing_tests_not_executed_by_this_script": [
            "solver/tests/verification/test_stochastic.py:333 (Green-kernel conditional oracle)",
            "solver/tests/verification/test_integrators.py:1622 (coupled RK4 analytic convergence)",
            "solver/tests/verification/test_integrators.py:3127 (dense charge local convergence)",
            "solver/tests/scenarios/test_brownian_run.py:814 (finite-depth first-passage stabilization)",
        ],
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "ou_observed_orders": payload["ou_covariance"]["observed_orders"],
        "philox_all_exact_match": payload["philox_known_answers"]["all_exact_match"],
        "dense_kinematic_difference": payload["rk4_dense_semantics"]["kinematic_interior_difference"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
