"""Independent OU Green-kernel quadrature shared by numerical and public tests."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np


def quadrature_covariance(
    argument: float,
    thermal_velocity_variance: float,
    rate: float,
) -> np.ndarray:
    """Integrate the OU Green-function kernel without production formulas."""

    nodes, weights = np.polynomial.legendre.leggauss(16)
    segment_count = max(1, math.ceil(argument / 0.25))
    edges = np.linspace(0.0, argument, segment_count + 1)
    integrals = np.zeros(3, dtype=np.float64)
    for lower, upper in pairwise(edges):
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        sample = midpoint + half_width * nodes
        decay = np.exp(-sample)
        displacement = -np.expm1(-sample)
        integrals += half_width * np.asarray(
            [
                np.dot(weights, 2.0 * displacement * displacement),
                np.dot(weights, 2.0 * displacement * decay),
                np.dot(weights, 2.0 * decay * decay),
            ]
        )
    return np.asarray(
        [
            thermal_velocity_variance * integrals[0] / rate**2,
            thermal_velocity_variance * integrals[1] / rate,
            thermal_velocity_variance * integrals[2],
        ]
    )


def covariance_matrix(components: np.ndarray) -> np.ndarray:
    return np.asarray(
        [[components[0], components[1]], [components[1], components[2]]],
        dtype=np.float64,
    )


def folded_gaussian_joint_cdf(
    position_limits: np.ndarray,
    velocity_limits: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    quadrature_order: int = 64,
) -> np.ndarray:
    """Law of (abs(X), sign(X)*V) for a nonsingular joint Gaussian.

    A flat specular wall and zero normal flow/forcing preserve the mirrored
    kinetic OU equation. Thus this is a continuous reflecting-OU oracle,
    including repeated impacts, not a production path or event construction.
    Integrate the conditional Gaussian velocity over both position preimages.
    The caller refines this independent quadrature separately from MC error.
    """

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    sigma_x = math.sqrt(float(covariance[0, 0]))
    slope = float(covariance[0, 1] / covariance[0, 0])
    conditional_sigma = math.sqrt(float(covariance[1, 1] - slope * covariance[0, 1]))
    result = np.empty_like(position_limits, dtype=np.float64)
    for index, (limit_x, limit_v) in enumerate(zip(position_limits, velocity_limits, strict=True)):
        magnitude = 0.5 * limit_x * (nodes + 1.0)
        integral = 0.0
        for sign in (-1.0, 1.0):
            position = sign * magnitude
            density = np.exp(-0.5 * ((position - mean[0]) / sigma_x) ** 2)
            density /= math.sqrt(2.0 * math.pi) * sigma_x
            conditional_mean = mean[1] + slope * (position - mean[0])
            argument = (limit_v - sign * conditional_mean) / conditional_sigma
            probability = np.asarray(
                [0.5 * math.erfc(-value / math.sqrt(2.0)) for value in argument]
            )
            integral += 0.5 * limit_x * float(np.dot(weights, density * probability))
        result[index] = integral
    return result


def time_linear_ou_moments(
    rate0: float,
    rate_slope: float,
    thermal: float,
    duration: float,
    position0: np.ndarray,
    velocity0: np.ndarray,
    flow0: np.ndarray,
    flow_slope: np.ndarray,
    *,
    quadrature_order: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate nonautonomous Green kernels, not frozen-step OU updates."""

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    time = 0.5 * duration * (nodes + 1.0)
    weight = 0.5 * duration * weights
    remaining = duration - time
    inner_time = time[:, None] + 0.5 * remaining[:, None] * (nodes[None, :] + 1.0)
    exponent = -rate0 * (
        inner_time - time[:, None] + 0.5 * rate_slope * (inner_time**2 - time[:, None] ** 2)
    )
    position_kernel = 0.5 * remaining * (np.exp(exponent) @ weights)
    terminal_kernel = np.exp(
        -rate0 * (duration - time + 0.5 * rate_slope * (duration**2 - time**2))
    )
    initial_kernel = np.exp(-rate0 * (time + 0.5 * rate_slope * time**2))
    rate = rate0 * (1.0 + rate_slope * time)
    forcing = rate[:, None] * (flow0 + time[:, None] * flow_slope)
    mean_position = (
        position0
        + velocity0 * np.dot(weight, initial_kernel)
        + (weight * position_kernel) @ forcing
    )
    mean_velocity = (
        velocity0 * math.exp(-rate0 * (duration + 0.5 * rate_slope * duration**2))
        + (weight * terminal_kernel) @ forcing
    )
    diffusion_weight = weight * (2.0 * thermal * rate)
    covariance = covariance_matrix(
        np.asarray(
            [
                np.dot(diffusion_weight, position_kernel**2),
                np.dot(diffusion_weight, position_kernel * terminal_kernel),
                np.dot(diffusion_weight, terminal_kernel**2),
            ]
        )
    )
    return mean_position, mean_velocity, covariance


def spatial_affine_ou_moments(
    rate: float,
    flow_gradient: float,
    thermal: float,
    duration: float,
    initial_state: np.ndarray,
    flow_intercept: float,
    *,
    quadrature_order: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact linear-SDE mean and Green covariance for u(x)=u0+b*x.

    The independent 2x2 exponential solves dx=v*dt and
    dv=rate*(u0+b*x-v)*dt+sqrt(2*rate*thermal)*dW.  It does not
    use the production predictor, frozen-root updates, or covariance helper.
    """

    matrix = np.asarray([[0.0, 1.0], [rate * flow_gradient, -rate]])
    half_trace = -0.5 * rate
    centered = matrix - half_trace * np.eye(2)
    discriminant = 0.25 * rate**2 + rate * flow_gradient

    def exponential(time: np.ndarray) -> np.ndarray:
        if discriminant > 0.0:
            frequency = math.sqrt(discriminant)
            even = np.cosh(frequency * time)
            odd = np.sinh(frequency * time) / frequency
        elif discriminant < 0.0:
            frequency = math.sqrt(-discriminant)
            even = np.cos(frequency * time)
            odd = np.sin(frequency * time) / frequency
        else:
            even = np.ones_like(time)
            odd = time
        return np.exp(half_trace * time)[:, None, None] * (
            even[:, None, None] * np.eye(2) + odd[:, None, None] * centered
        )

    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    time = 0.5 * duration * (nodes + 1.0)
    weight = 0.5 * duration * weights
    velocity_impulse = exponential(time)[:, :, 1]
    mean = exponential(np.asarray([duration]))[0] @ initial_state
    mean += rate * flow_intercept * (weight @ velocity_impulse)
    covariance = (2.0 * rate * thermal) * ((velocity_impulse.T * weight) @ velocity_impulse)
    return mean, covariance


def hermite_ou_frame_moments(
    rate: float,
    thermal: float,
    macro_duration: float,
    query_time: float,
    depth: int,
    initial_state: np.ndarray,
    target_velocity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent Gaussian law of the finite Hermite leaf at a fixed query.

    Exact OU endpoint cross covariance comes from independent Green integrals
    and deterministic transition matrices. Cubic Hermite position and its
    derivative are then linear maps of those endpoints. This law need not equal
    the continuous OU law inside a leaf, particularly for velocity.
    """

    leaf_duration = macro_duration / 2**depth
    leaf_index = math.floor(query_time / leaf_duration)
    left_time = leaf_index * leaf_duration
    right_time = left_time + leaf_duration
    phase = (query_time - left_time) / leaf_duration

    def moments(time: float) -> tuple[np.ndarray, np.ndarray]:
        decay = math.exp(-rate * time)
        displacement = -math.expm1(-rate * time) / rate
        mean = np.asarray(
            [
                initial_state[0]
                + target_velocity * time
                + (initial_state[1] - target_velocity) * displacement,
                target_velocity + (initial_state[1] - target_velocity) * decay,
            ]
        )
        return mean, covariance_matrix(quadrature_covariance(rate * time, thermal, rate))

    left_mean, left_covariance = moments(left_time)
    right_mean, right_covariance = moments(right_time)
    transition = np.asarray(
        [[1.0, -math.expm1(-rate * leaf_duration) / rate], [0.0, math.exp(-rate * leaf_duration)]]
    )
    cross = left_covariance @ transition.T
    covariance = np.block([[left_covariance, cross], [cross.T, right_covariance]])
    s = phase
    mapping = np.asarray(
        [
            [
                2 * s**3 - 3 * s**2 + 1,
                leaf_duration * (s**3 - 2 * s**2 + s),
                -2 * s**3 + 3 * s**2,
                leaf_duration * (s**3 - s**2),
            ],
            [
                (6 * s**2 - 6 * s) / leaf_duration,
                3 * s**2 - 4 * s + 1,
                (-6 * s**2 + 6 * s) / leaf_duration,
                3 * s**2 - 2 * s,
            ],
        ]
    )
    return mapping @ np.concatenate((left_mean, right_mean)), mapping @ covariance @ mapping.T


def stationary_oml_forced_ou_mean(
    *,
    duration: float,
    rate: float,
    initial_charge: float,
    radius: float,
    density: float,
    electron_temperature: float,
    ion_temperature: float,
    ion_mass: float,
    particle_mass: float,
    electric_field: np.ndarray,
    initial_velocity: np.ndarray,
    flow: np.ndarray,
    intervals: int = 4096,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Independent negative-branch scalar OML ODE plus OU Green convolution.

    SI constants and absorbing-sphere flux/capacitance equations are written
    here; no production charging, stage, or state updater is called. RK4 of the
    scalar charge followed by composite Simpson Green quadrature is refined
    independently. Only stationary, velocity-independent OML is covered.
    """

    elementary_charge = 1.602176634e-19
    boltzmann = 1.380649e-23
    epsilon = 8.8541878128e-12
    electron_mass = 9.1093837139e-31
    debye = math.sqrt(
        epsilon
        * boltzmann
        / (elementary_charge**2 * density)
        / (1.0 / electron_temperature + 1.0 / ion_temperature)
    )
    capacitance = 4.0 * math.pi * epsilon * radius * (1.0 + radius / debye)
    electron_voltage = boltzmann * electron_temperature / elementary_charge
    ion_voltage = boltzmann * ion_temperature / elementary_charge
    electron_rate = (
        math.pi
        * radius**2
        * density
        * math.sqrt(8.0 * boltzmann * electron_temperature / (math.pi * electron_mass))
    )
    ion_rate = (
        math.pi
        * radius**2
        * density
        * math.sqrt(8.0 * boltzmann * ion_temperature / (math.pi * ion_mass))
    )

    def derivative(charge: float) -> float:
        potential = elementary_charge * charge / capacitance
        return ion_rate * (1.0 - potential / ion_voltage) - electron_rate * math.exp(
            potential / electron_voltage
        )

    time = np.linspace(0.0, duration, intervals + 1)
    dt = duration / intervals
    charge = np.empty(intervals + 1)
    charge[0] = initial_charge
    for index in range(intervals):
        start = charge[index]
        k1 = derivative(start)
        k2 = derivative(start + 0.5 * dt * k1)
        k3 = derivative(start + 0.5 * dt * k2)
        k4 = derivative(start + dt * k3)
        charge[index + 1] = start + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    if intervals % 2 or bool((charge > 0.0).any()):
        raise ValueError("reference requires even intervals and a maintained negative OML branch")
    weights = np.full(intervals + 1, 2.0)
    weights[1::2] = 4.0
    weights[[0, -1]] = 1.0
    weights *= dt / 3.0
    decay = np.exp(-rate * (duration - time))
    displacement = -np.expm1(-rate * duration) / rate
    coupling = elementary_charge * electric_field / particle_mass
    mean_position = flow * duration + (initial_velocity - flow) * displacement
    mean_velocity = flow + (initial_velocity - flow) * math.exp(-rate * duration)
    mean_position += coupling * np.dot(weights, charge * (1.0 - decay) / rate)
    mean_velocity += coupling * np.dot(weights, charge * decay)
    return float(charge[-1]), mean_position, mean_velocity
