"""Qualify explicit Maxwell-sphere regimes against independent collision references.

Run with ``uv run --locked python -m tools.vv.gas_collision.qualify``.
Input CSV columns describe SI physical regimes, never a COMSOL producer schema.
Output is external V&V evidence, not a new production SDE or trajectory engine.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

from chamber_particles.physics.forces import epstein_finite_speed_relaxation

from .collision_reference import (
    BOLTZMANN_J_K,
    CollisionMoments,
    SphereBath,
    collision_moments,
    compound_poisson_cumulants,
    sample_collision_jumps,
    sample_frozen_collision_increments,
)


@dataclass(frozen=True, slots=True)
class Regime:
    label: str
    bath: SphereBath
    speed_ratio: float
    gas_mean_free_path_m: float


def read_regimes(path: Path) -> list[Regime]:
    """Read concrete physical cases, without inferring units or surface parameters."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        regimes = [
            Regime(
                row["label"],
                SphereBath(
                    particle_mass_kg=float(row["particle_mass_kg"]),
                    molecular_mass_kg=float(row["molecular_mass_kg"]),
                    gas_density_kg_m3=float(row["gas_density_kg_m3"]),
                    gas_temperature_K=float(row["gas_temperature_K"]),
                    diameter_m=float(row["diameter_m"]),
                    diffuse_fraction=float(row["diffuse_fraction"]),
                    surface_temperature_K=float(row["surface_temperature_K"]),
                ),
                float(row["speed_ratio"]),
                float(row["gas_mean_free_path_m"]),
            )
            for row in reader
        ]
    if not regimes or len({regime.label for regime in regimes}) != len(regimes):
        raise ValueError("regime labels must be nonempty and unique")
    return regimes


def production_mean(regime: Regime) -> tuple[float, float]:
    """Evaluate the current pure supported mean law on the identical bath and speed."""
    bath = regime.bath
    speed = math.sqrt(2.0) * bath.thermal_speed_m_s * regime.speed_ratio
    evaluation = epstein_finite_speed_relaxation(
        mass_kg=np.asarray([bath.particle_mass_kg]),
        drag_diameter_m=np.asarray([bath.diameter_m]),
        velocity_m_s=np.asarray([[speed, 0.0]]),
        gas_velocity_m_s=np.zeros((1, 2)),
        gas_density_kg_m3=np.asarray([bath.gas_density_kg_m3]),
        gas_temperature_K=np.asarray([bath.gas_temperature_K]),
        gas_mean_free_path_m=np.asarray([regime.gas_mean_free_path_m]),
        gas_molecular_mass_kg=bath.molecular_mass_kg,
        diffuse_reflection_fraction=bath.diffuse_fraction,
        maximum_speed_ratio=max(1.0, 2.0 * regime.speed_ratio),
    )
    if not bool(evaluation.applicable[0]):
        raise ValueError("mean-law comparison must lie inside the unchanged free-molecular gate")
    return -float(evaluation.rate_s_inv[0]) * speed, float(evaluation.rate_s_inv[0])


def _linear_rate(bath: SphereBath) -> float:
    cbar = math.sqrt(8.0 / math.pi) * bath.thermal_speed_m_s
    beta = 4.0 * math.pi / 3.0 * (0.5 * bath.diameter_m) ** 2 * bath.gas_density_kg_m3 * cbar
    return beta * (1.0 + bath.diffuse_fraction * math.pi / 8.0) / bath.particle_mass_kg


def compare_moments(regime: Regime, orders: list[int]) -> tuple[dict[str, Any], CollisionMoments]:
    """Separate production mean agreement, quadrature error and proposed-noise differences."""
    bath = regime.bath
    estimates = [collision_moments(bath, regime.speed_ratio, gas_order=order) for order in orders]
    reference = estimates[-1]
    preceding = estimates[-2]
    mean, instantaneous_rate = production_mean(regime)
    linear_rate = _linear_rate(bath)
    thermal_variance = BOLTZMANN_J_K * bath.gas_temperature_K / bath.particle_mass_kg
    linear_fdt = 2.0 * linear_rate * thermal_variance
    instantaneous_fdt = 2.0 * instantaneous_rate * thermal_variance
    mean_scale = max(abs(mean), linear_rate * bath.thermal_speed_m_s)
    covariance = np.diag(reference.covariance_rate_m2_s3)
    recoil = collision_moments(bath, regime.speed_ratio, gas_order=orders[-1], specular_recoil=True)
    row: dict[str, Any] = {
        "label": regime.label,
        "speed_ratio_S": regime.speed_ratio,
        "mass_ratio": bath.mass_ratio,
        "particle_mass_kg": bath.particle_mass_kg,
        "gas_temperature_K": bath.gas_temperature_K,
        "diffuse_fraction": bath.diffuse_fraction,
        "surface_to_gas_temperature_ratio": bath.surface_temperature_K / bath.gas_temperature_K,
        "production_mean_acceleration_m_s2": mean,
        "collision_mean_acceleration_m_s2": float(reference.acceleration_m_s2[0]),
        "mean_residual_normalized": abs(mean - reference.acceleration_m_s2[0]) / mean_scale,
        "gas_quadrature_mean_change_normalized": abs(
            reference.acceleration_m_s2[0] - preceding.acceleration_m_s2[0]
        )
        / mean_scale,
        "gas_quadrature_Q_change_relative_max": float(
            np.max(
                np.abs(
                    np.diag(reference.covariance_rate_m2_s3 - preceding.covariance_rate_m2_s3)
                    / covariance
                )
            )
        ),
        "Q_parallel_m2_s3": float(covariance[0]),
        "Q_transverse_m2_s3": float(covariance[1]),
        "Q_parallel_over_transverse": float(covariance[0] / covariance[1]),
        "Q_parallel_over_linear_FDT": float(covariance[0] / linear_fdt),
        "Q_transverse_over_linear_FDT": float(covariance[1] / linear_fdt),
        "Q_parallel_over_instantaneous_rate_FDT": float(covariance[0] / instantaneous_fdt),
        "Q_transverse_over_instantaneous_rate_FDT": float(covariance[1] / instantaneous_fdt),
        "collision_rate_s_inv": reference.collision_rate_s_inv,
        "incident_probability_mass": reference.incident_probability_mass,
        "linear_relaxation_rate_s_inv": linear_rate,
        "instantaneous_relaxation_rate_s_inv": instantaneous_rate,
        "specular_only_recoil_mean_change_normalized": abs(
            recoil.acceleration_m_s2[0] - reference.acceleration_m_s2[0]
        )
        / mean_scale,
        "specular_only_recoil_Q_change_relative_max": float(
            np.max(
                np.abs(
                    np.diag(recoil.covariance_rate_m2_s3 - reference.covariance_rate_m2_s3)
                    / covariance
                )
            )
        ),
    }
    return row, reference


def _characteristic_probe(jumps: np.ndarray, width_m_s: float) -> dict[str, Any]:
    """Test the exact frozen jump generator against its second-order truncation."""
    scaled = jumps / width_m_s
    exact_samples = np.exp(1j * scaled) - 1.0
    exact = np.mean(exact_samples, axis=0)
    second_order = 1j * np.mean(scaled, axis=0) - 0.5 * np.mean(scaled**2, axis=0)
    residual = exact - second_order
    paired_residual = exact_samples - 1j * scaled + 0.5 * scaled**2
    return {
        "width_m_s": width_m_s,
        "exact_generator_per_collision_real": exact.real.tolist(),
        "exact_generator_per_collision_imag": exact.imag.tolist(),
        "second_order_generator_per_collision_real": second_order.real.tolist(),
        "second_order_generator_per_collision_imag": second_order.imag.tolist(),
        "absolute_truncation_residual_per_collision": np.abs(residual).tolist(),
        "relative_truncation_residual": (np.abs(residual) / np.abs(second_order)).tolist(),
        "exact_generator_sample_standard_error": (
            np.std(exact_samples, axis=0, ddof=1) / math.sqrt(jumps.shape[0])
        ).tolist(),
        "truncation_residual_paired_standard_error": (
            np.std(paired_residual, axis=0, ddof=1) / math.sqrt(jumps.shape[0])
        ).tolist(),
    }


def _frozen_distribution_probe(
    regime: Regime, reference: CollisionMoments, duration_s: float, count: int, seed: int
) -> dict[str, Any]:
    increments = sample_frozen_collision_increments(
        regime.bath, regime.speed_ratio, reference.collision_rate_s_inv, duration_s, count, seed
    )
    mean = reference.acceleration_m_s2 * duration_s
    standard_deviation = np.sqrt(np.diag(reference.covariance_rate_m2_s3) * duration_s)
    standardized = (increments - mean) / standard_deviation
    lower_tail = np.mean(standardized < -3.0, axis=0)
    upper_tail = np.mean(standardized > 3.0, axis=0)
    return {
        "duration_s": duration_s,
        "intervals": count,
        "seed": seed,
        "no_collision_atom_probability": math.exp(-reference.collision_rate_s_inv * duration_s),
        "Gaussian_no_collision_atom_probability": 0.0,
        "Gaussian_each_3sigma_tail_probability": NormalDist().cdf(-3.0),
        "compound_Poisson_lower_3sigma_tail_probability": lower_tail.tolist(),
        "compound_Poisson_upper_3sigma_tail_probability": upper_tail.tolist(),
        "lower_tail_sampling_standard_error": np.sqrt(
            lower_tail * (1.0 - lower_tail) / count
        ).tolist(),
        "upper_tail_sampling_standard_error": np.sqrt(
            upper_tail * (1.0 - upper_tail) / count
        ).tolist(),
        "lower_tail_Wilson_95_interval": _binomial_wilson_interval(lower_tail, count),
        "upper_tail_Wilson_95_interval": _binomial_wilson_interval(upper_tail, count),
        "zero_count_interpretation": "Zero observed tail counts do not establish zero probability; use the reported Wilson interval.",
        "scope": "Frozen bath/velocity distribution only; this is not a wall crossing or state-dependent trajectory calculation.",
    }


def _binomial_wilson_interval(probability: np.ndarray, count: int) -> list[list[float]]:
    z = NormalDist().inv_cdf(0.975)
    denominator = 1.0 + z**2 / count
    center = (probability + z**2 / (2.0 * count)) / denominator
    half = z * np.sqrt(probability * (1.0 - probability) / count + z**2 / (4.0 * count**2))
    half /= denominator
    return np.stack(
        (np.maximum(0.0, center - half), np.minimum(1.0, center + half)), axis=1
    ).tolist()


def qualify_jumps(
    regime: Regime,
    reference: CollisionMoments,
    count: int,
    seed: int,
    durations: list[float],
    interval_count: int,
) -> dict[str, Any]:
    """Check the independent sampler and measure finite frozen-increment non-Gaussianity."""
    start = time.perf_counter()
    jumps = sample_collision_jumps(regime.bath, regime.speed_ratio, count, seed)
    sample_mean = np.mean(jumps, axis=0)
    sample_second = np.mean(jumps**2, axis=0)
    standard_error_mean = np.std(jumps, axis=0, ddof=1) / math.sqrt(count)
    standard_error_second = np.std(jumps**2, axis=0, ddof=1) / math.sqrt(count)
    predicted_mean = reference.acceleration_m_s2 / reference.collision_rate_s_inv
    predicted_second = np.diag(reference.covariance_rate_m2_s3) / reference.collision_rate_s_inv
    thermal_width = math.sqrt(
        BOLTZMANN_J_K * regime.bath.gas_temperature_K / regime.bath.particle_mass_kg
    )
    cumulants = {
        str(duration): compound_poisson_cumulants(jumps, reference.collision_rate_s_inv, duration)
        for duration in durations
    }
    return {
        "label": regime.label,
        "seed": seed,
        "samples": count,
        "seconds": time.perf_counter() - start,
        "mean_error_in_estimated_standard_errors": (
            (sample_mean - predicted_mean) / standard_error_mean
        ).tolist(),
        "raw_second_error_in_estimated_standard_errors": (
            (sample_second - predicted_second) / standard_error_second
        ).tolist(),
        "rms_jump_over_particle_thermal_width": (np.sqrt(sample_second) / thermal_width).tolist(),
        "frozen_compound_Poisson_cumulants": cumulants,
        "characteristic_probe_particle_thermal_width": _characteristic_probe(jumps, thermal_width),
        "characteristic_probe_one_tenth_particle_thermal_width": _characteristic_probe(
            jumps, thermal_width / 10.0
        ),
        "frozen_distribution_probe": _frozen_distribution_probe(
            regime, reference, durations[0], interval_count, seed + 1000000
        ),
        "scope": "Frozen velocity/bath jump law only. Neither a nonlinear trajectory distribution nor a wall first-arrival test. Finite-recoil diffuse dynamics remain unqualified.",
    }


def run_qualification(
    regimes: list[Regime],
    orders: list[int],
    jump_labels: set[str],
    jump_count: int,
    durations: list[float],
    interval_count: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    jump_results: list[dict[str, Any]] = []
    for index, regime in enumerate(regimes):
        row, reference = compare_moments(regime, orders)
        rows.append(row)
        if regime.label in jump_labels:
            jump_results.append(
                qualify_jumps(
                    regime, reference, jump_count, 20261014 + index, durations, interval_count
                )
            )
    if jump_labels != {str(record["label"]) for record in jump_results}:
        raise ValueError("all requested jump labels must identify a supplied regime")
    return {
        "model": "3D local Maxwellian, sphere, heavy diffuse flux-Maxwell, prescribed Tw",
        "status": "COLLISION_MOMENT_QUALIFICATION_ONLY",
        "gas_orders": orders,
        "rows": rows,
        "jump_results": jump_results,
        "not_qualified": [
            "Nonlinear SDE trajectory law, multiplicative-noise integration, wall first arrival or deposition.",
            "Physically unconstrained 3D motion from a meridional 2DOF solver.",
            "Species-resolved mixture truth from one supplied pseudogas mass.",
            "Finite-recoil diffuse energy transfer or dynamic particle temperature.",
            "High-drift thermophoresis without an incident-distribution/heat-flux closure.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regimes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gas-orders", default="32,64,96,128")
    parser.add_argument("--jump-labels", default="")
    parser.add_argument("--jump-count", default=200000, type=int)
    parser.add_argument("--durations-s", required=True)
    parser.add_argument("--interval-count", default=50000, type=int)
    args = parser.parse_args()
    orders = [int(value) for value in args.gas_orders.split(",")]
    if len(orders) < 2 or min(orders) < 4 or orders != sorted(set(orders)):
        raise ValueError("supply at least two increasing distinct gas quadrature orders >=4")
    if args.jump_count < 2:
        raise ValueError("jump-count must be at least two")
    durations = [float(value) for value in args.durations_s.split(",")]
    if (
        not all(math.isfinite(value) and value > 0.0 for value in durations)
        or args.interval_count < 2
    ):
        raise ValueError("duration and interval count must be positive")
    start = time.perf_counter()
    tracemalloc.start()
    report = run_qualification(
        read_regimes(args.regimes),
        orders,
        {value for value in args.jump_labels.split(",") if value},
        args.jump_count,
        durations,
        args.interval_count,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    report["seconds"] = time.perf_counter() - start
    report["peak_traced_allocation_bytes"] = peak
    report["memory_scope"] = (
        "Python/NumPy allocations traced by tracemalloc; not whole-process RSS."
    )
    report["input_binding"] = {
        "path": str(args.regimes.resolve()),
        "sha256": hashlib.sha256(args.regimes.read_bytes()).hexdigest(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "qualification.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    with (args.output / "moment_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["rows"][0]))
        writer.writeheader()
        writer.writerows(report["rows"])
    print(
        json.dumps(
            {"rows": len(report["rows"]), "seconds": report["seconds"], "output": str(args.output)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
