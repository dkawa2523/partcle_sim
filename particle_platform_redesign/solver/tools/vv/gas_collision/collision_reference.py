"""Independent sphere collision quadrature and frozen collision-jump sampling.

The incident gas is a three-dimensional Maxwellian.  A hemisphere quadrature
acts on the actual specular/diffuse impulse, then axial symmetry reduces the
gas integral to shifted spherical speed and polar-angle quadrature.
This reference never uses the production finite-speed drag coefficient.

Diffuse reflection is the explicitly prescribed-temperature heavy-particle
flux-Maxwell approximation.  A recoil option changes ONLY the specular branch;
it does not turn diffuse reflection into a finite-recoil energy model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]

BOLTZMANN_J_K = 1.380649e-23
OUTGOING_NORMAL_CUTOFF = 10.0
INCIDENT_GAUSSIAN_CUTOFF = 12.0


@dataclass(frozen=True, slots=True)
class SphereBath:
    """SI authorities for one sphere and one prescribed Maxwellian bath."""

    particle_mass_kg: float
    molecular_mass_kg: float
    gas_density_kg_m3: float
    gas_temperature_K: float
    diameter_m: float
    diffuse_fraction: float
    surface_temperature_K: float

    def __post_init__(self) -> None:
        positive = (
            self.particle_mass_kg,
            self.molecular_mass_kg,
            self.gas_density_kg_m3,
            self.gas_temperature_K,
            self.diameter_m,
            self.surface_temperature_K,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("collision bath SI authorities must be finite and positive")
        if not math.isfinite(self.diffuse_fraction) or not 0.0 <= self.diffuse_fraction <= 1.0:
            raise ValueError("diffuse_fraction must be finite and in [0,1]")

    @property
    def thermal_speed_m_s(self) -> float:
        """Return sqrt(kBT/m), not the mean or most-probable speed."""
        return math.sqrt(BOLTZMANN_J_K * self.gas_temperature_K / self.molecular_mass_kg)

    @property
    def mass_ratio(self) -> float:
        return self.molecular_mass_kg / self.particle_mass_kg

    @property
    def rate_scale_s_inv(self) -> float:
        number_density = self.gas_density_kg_m3 / self.molecular_mass_kg
        return number_density * (0.5 * self.diameter_m) ** 2 * self.thermal_speed_m_s


@dataclass(frozen=True, slots=True)
class CollisionMoments:
    """First moment and covariance RATE of the three-dimensional jump law."""

    acceleration_m_s2: FloatArray
    covariance_rate_m2_s3: FloatArray
    collision_rate_s_inv: float
    incident_probability_mass: float


def outgoing_flux_moments(order: int = 64) -> tuple[float, float, float]:
    """Numerically integrate normal flux moments and Gaussian tangential variance."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    normal = 0.5 * OUTGOING_NORMAL_CUTOFF * (nodes + 1.0)
    measure = 0.5 * OUTGOING_NORMAL_CUTOFF * weights * normal * np.exp(-0.5 * normal**2)
    gaussian_nodes, gaussian_weights = np.polynomial.hermite.hermgauss(4)
    tangent_second = float(np.sum(gaussian_weights * 2.0 * gaussian_nodes**2) / math.sqrt(math.pi))
    return (
        float(np.sum(measure * normal)),
        float(np.sum(measure * normal**2)),
        tangent_second,
    )


def surface_moment_coefficients(
    bath: SphereBath,
    *,
    normal_order: int = 24,
    azimuth_count: int = 16,
    outgoing_order: int = 64,
    specular_recoil: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Integrate actual hemisphere impulse; coefficients multiply |g| powers 1,2,3."""
    nodes, weights = np.polynomial.legendre.leggauss(normal_order)
    cosine = 0.5 * (nodes - 1.0)
    azimuth = 2.0 * math.pi * np.arange(azimuth_count) / azimuth_count
    transverse = np.sqrt(1.0 - cosine**2)
    normals = np.stack(
        np.broadcast_arrays(
            cosine[:, None],
            transverse[:, None] * np.cos(azimuth)[None, :],
            transverse[:, None] * np.sin(azimuth)[None, :],
        ),
        axis=-1,
    )
    measure = np.broadcast_to(
        -cosine[:, None] * 0.5 * weights[:, None] * 2.0 * math.pi / azimuth_count,
        normals.shape[:2],
    )
    alpha = bath.mass_ratio
    specular_alpha = alpha / (1.0 + alpha) if specular_recoil else alpha
    specular_impulse = 2.0 * specular_alpha * cosine[:, None, None] * normals
    specular_mean = np.sum(measure * specular_impulse[:, :, 0])
    specular_second = np.sum(measure[:, :, None] * specular_impulse**2, axis=(0, 1))
    outgoing_mean, outgoing_normal_second, outgoing_tangent_second = outgoing_flux_moments(
        outgoing_order
    )
    temperature_ratio = bath.surface_temperature_K / bath.gas_temperature_K
    mean_outgoing_x = math.sqrt(temperature_ratio) * outgoing_mean * normals[:, :, 0]
    outgoing_second = temperature_ratio * (
        outgoing_tangent_second + (outgoing_normal_second - outgoing_tangent_second) * normals**2
    )
    sigma = bath.diffuse_fraction
    mean_coefficients = np.asarray(
        [
            -sigma * alpha * np.sum(measure * mean_outgoing_x),
            (1.0 - sigma) * specular_mean + sigma * alpha * np.sum(measure),
            0.0,
        ]
    )
    parallel_coefficients = np.asarray(
        [
            sigma * alpha**2 * np.sum(measure * outgoing_second[:, :, 0]),
            -2.0 * sigma * alpha**2 * np.sum(measure * mean_outgoing_x),
            (1.0 - sigma) * specular_second[0] + sigma * alpha**2 * np.sum(measure),
        ]
    )
    transverse_coefficients = np.asarray(
        [
            sigma * alpha**2 * np.sum(measure * outgoing_second[:, :, 1]),
            0.0,
            (1.0 - sigma) * specular_second[1],
        ]
    )
    return mean_coefficients, parallel_coefficients, transverse_coefficients


def collision_moments(
    bath: SphereBath,
    speed_ratio: float,
    *,
    gas_order: int = 96,
    normal_order: int = 24,
    specular_recoil: bool = False,
) -> CollisionMoments:
    """Integrate the full 3D Maxwell bath with velocity along the first component."""
    if not math.isfinite(speed_ratio) or speed_ratio < 0.0:
        raise ValueError("speed_ratio S=|v-u|/sqrt(2kBT/m) must be finite and nonnegative")
    nodes, weights = np.polynomial.legendre.leggauss(gas_order)
    drift = math.sqrt(2.0) * speed_ratio
    lower = max(0.0, drift - INCIDENT_GAUSSIAN_CUTOFF)
    upper = drift + INCIDENT_GAUSSIAN_CUTOFF
    radial = 0.5 * (upper - lower) * nodes + 0.5 * (upper + lower)
    radial_weights = 0.5 * (upper - lower) * weights
    speed = np.broadcast_to(radial[:, None], (gas_order, gas_order))
    cosine = nodes[None, :]
    exponent = (speed - drift) ** 2 + 2.0 * speed * drift * (1.0 + cosine)
    measure = (
        radial_weights[:, None]
        * weights[None, :]
        * speed**2
        * np.exp(-0.5 * exponent)
        / math.sqrt(2.0 * math.pi)
    )
    mean_coefficients, parallel_coefficients, transverse_coefficients = surface_moment_coefficients(
        bath, normal_order=normal_order, specular_recoil=specular_recoil
    )
    powers = np.stack((speed, speed**2, speed**3), axis=-1)
    mean_parallel = np.sum(powers * mean_coefficients, axis=-1)
    local_parallel = np.sum(powers * parallel_coefficients, axis=-1)
    local_transverse = np.sum(powers * transverse_coefficients, axis=-1)
    direction_parallel_squared = cosine**2
    difference = local_parallel - local_transverse
    covariance_parallel = np.sum(
        measure * (local_transverse + difference * direction_parallel_squared)
    )
    covariance_transverse = np.sum(
        measure * (local_transverse + 0.5 * difference * (1.0 - direction_parallel_squared))
    )
    acceleration = np.asarray([np.sum(measure * mean_parallel * cosine), 0.0, 0.0])
    covariance = np.diag([covariance_parallel, covariance_transverse, covariance_transverse])
    return CollisionMoments(
        acceleration * bath.rate_scale_s_inv * bath.thermal_speed_m_s,
        covariance * bath.rate_scale_s_inv * bath.thermal_speed_m_s**2,
        float(math.pi * bath.rate_scale_s_inv * np.sum(measure * speed)),
        float(np.sum(measure)),
    )


def _tangent_frame(unit_normal: FloatArray) -> tuple[FloatArray, FloatArray]:
    axis = np.eye(3)[np.argmin(np.abs(unit_normal), axis=1)]
    first = np.cross(unit_normal, axis)
    first /= np.linalg.norm(first, axis=1)[:, None]
    return first, np.cross(unit_normal, first)


def _incident_velocity_proposal(rng: np.random.Generator, count: int, drift: float) -> FloatArray:
    """Proposal density proportional to (|v|+|c|)f(c), with an exact Gaussian mixture."""
    velocity = rng.normal(size=(count, 3))
    speed_weighted = rng.random(count) >= drift / (drift + math.sqrt(8.0 / math.pi))
    selected = velocity[speed_weighted]
    direction = selected / np.linalg.norm(selected, axis=1)[:, None]
    radii = np.sqrt(2.0 * rng.gamma(2.0, size=selected.shape[0]))
    velocity[speed_weighted] = direction * radii[:, None]
    return velocity


def flux_weighted_incident_velocities(
    rng: np.random.Generator, count: int, speed_ratio: float
) -> FloatArray:
    """Exact rejection sampling of |c-v|f(c), without a false Gaussian upper bound."""
    drift = math.sqrt(2.0) * speed_ratio
    accepted: list[FloatArray] = []
    remaining = count
    while remaining:
        proposed = _incident_velocity_proposal(rng, min(65536, 2 * remaining + 32), drift)
        relative = proposed - np.asarray([drift, 0.0, 0.0])
        acceptance = np.linalg.norm(relative, axis=1) / (drift + np.linalg.norm(proposed, axis=1))
        retained = relative[rng.random(proposed.shape[0]) < acceptance][:remaining]
        accepted.append(retained)
        remaining -= retained.shape[0]
    return np.concatenate(accepted)


def sample_collision_jumps(
    bath: SphereBath,
    speed_ratio: float,
    count: int,
    seed: int,
    *,
    specular_recoil: bool = False,
) -> FloatArray:
    """Draw conditional collision impulses in m/s; this is not a trajectory engine."""
    if count < 1 or not math.isfinite(speed_ratio) or speed_ratio < 0.0:
        raise ValueError("jump count must be positive and speed ratio finite/nonnegative")
    rng = np.random.default_rng(seed)
    relative = flux_weighted_incident_velocities(rng, count, speed_ratio)
    direction = relative / np.linalg.norm(relative, axis=1)[:, None]
    tangent_first, tangent_second = _tangent_frame(direction)
    cosine = np.sqrt(rng.random(count))
    azimuth = 2.0 * math.pi * rng.random(count)
    transverse = np.sqrt(1.0 - cosine**2)
    normal = (
        -cosine[:, None] * direction
        + (transverse * np.cos(azimuth))[:, None] * tangent_first
        + (transverse * np.sin(azimuth))[:, None] * tangent_second
    )
    alpha = bath.mass_ratio
    specular_alpha = alpha / (1.0 + alpha) if specular_recoil else alpha
    projection = np.sum(relative * normal, axis=1)
    jumps = 2.0 * specular_alpha * projection[:, None] * normal
    diffuse = rng.random(count) < bath.diffuse_fraction
    selected_normal = normal[diffuse]
    first, second = _tangent_frame(selected_normal)
    normal_speed = np.sqrt(-2.0 * np.log(1.0 - rng.random(selected_normal.shape[0])))
    tangential_speed = rng.normal(size=(selected_normal.shape[0], 2))
    outgoing = (
        normal_speed[:, None] * selected_normal
        + tangential_speed[:, :1] * first
        + tangential_speed[:, 1:] * second
    ) * math.sqrt(bath.surface_temperature_K / bath.gas_temperature_K)
    jumps[diffuse] = alpha * (relative[diffuse] - outgoing)
    return jumps * bath.thermal_speed_m_s


def compound_poisson_cumulants(
    jumps_m_s: FloatArray, collision_rate_s_inv: float, duration_s: float
) -> dict[str, list[float] | float]:
    """Exact frozen-jump cumulant identities, estimated by the conditional jump sample."""
    expected_count = collision_rate_s_inv * duration_s
    variance = expected_count * np.mean(jumps_m_s**2, axis=0)
    third = expected_count * np.mean(jumps_m_s**3, axis=0)
    fourth = expected_count * np.mean(jumps_m_s**4, axis=0)
    return {
        "expected_collisions": expected_count,
        "mean_increment_m_s": (expected_count * np.mean(jumps_m_s, axis=0)).tolist(),
        "increment_variance_m2_s2": variance.tolist(),
        "standardized_skewness": (third / variance**1.5).tolist(),
        "excess_kurtosis": (fourth / variance**2).tolist(),
    }


def sample_frozen_collision_increments(
    bath: SphereBath,
    speed_ratio: float,
    collision_rate_s_inv: float,
    duration_s: float,
    intervals: int,
    seed: int,
) -> FloatArray:
    """Draw a frozen compound Poisson increment, with no state/field evolution."""
    if duration_s <= 0.0 or intervals < 1:
        raise ValueError("frozen interval duration/count must be positive")
    counts = np.random.default_rng(seed).poisson(collision_rate_s_inv * duration_s, intervals)
    increments = np.zeros((intervals, 3))
    for start in range(0, intervals, 2048):
        stop = min(start + 2048, intervals)
        interval_index = np.repeat(np.arange(stop - start), counts[start:stop])
        if not interval_index.size:
            continue
        collision_seed = int(np.random.SeedSequence([seed, 1, start]).generate_state(1)[0])
        jumps = sample_collision_jumps(bath, speed_ratio, interval_index.size, collision_seed)
        for component in range(3):
            increments[start:stop, component] = np.bincount(
                interval_index, weights=jumps[:, component], minlength=stop - start
            )
    return increments
